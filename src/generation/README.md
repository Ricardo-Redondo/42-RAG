# `generation/` — from a question + its sources to an answer

This folder turns retrieved sources into a natural-language answer with
`Qwen/Qwen3-0.6B`. It runs in the `answer` and `answer_dataset` commands.

```
Pipeline.answer          → retrieve (k sources) → Generator.answer
Pipeline.answer_dataset  → read a search-results file → Generator.answer

Generator.answer(search_results)               batch in → batch out
 ├─ load()                    only if at least one question has sources
 └─ for each question (tqdm):
      generate(result)
       ├─ no sources?         → NO_SOURCES (model never called)
       ├─ select_context()    pick + trim the sources for the prompt
       │    └─ source_text()  file + offsets → text (cached)
       ├─ build_messages()    system rules + numbered sources + question
       ├─ apply_chat_template → tokenizer → generate → slice → decode
       └─ add_citations()     clean up + add the "Sources:" list
```

Running example below: *"What HTTP endpoint is used to dynamically load a LoRA
adapter in vLLM?"* with `answer "..." --k 10`. Values in comments are typical
examples.

---

## 1. The two libraries

| Library | Role |
|---|---|
| **`transformers`** (Hugging Face) | the "driver": downloads Qwen, builds the network, gives the tokenizer, the chat template and the `generate` loop |
| **`torch`** (PyTorch) | the "engine": the maths on tensors (big arrays of numbers) that runs the model |

Both are imported **inside methods** (`load`, `generate`), never at the top of
the file: importing them takes 3–7 s, which `search` and `index` would
otherwise pay too. For type hints they're imported under `TYPE_CHECKING`
(read by mypy, skipped at runtime) together with
`from __future__ import annotations` (annotations stay strings, so the
missing import doesn't crash).

---

## 2. Constants

```python
NO_SOURCES = "No relevant sources were found to answer this question."    # k=0 / nothing found
NOT_IN_SOURCES = "The provided sources do not contain the answer."         # the refusal sentence

SYSTEM_PROMPT = (
    "You are a technical assistant for the vLLM codebase. "
    "Answer the question using ONLY the numbered sources given by the user.\n"
    "1. Answer directly in 2 to 4 sentences ... name the subject ..."   # eval: self-contained
    "2. After each fact, cite ... like [1]."                           # eval: source-grounded
    "3. Only state what the sources say. Never invent ..."             # eval: faithful
    "4. Do not describe or list the sources ..."                       # no slow rambling
    "5. Only if none of the sources answers ... reply ... NOT_IN_SOURCES"  # faithful when retrieval failed
)

CITATION = re.compile(r"\[(\d+)\]")
# \[ … \]  literal brackets (escaped: [ ] mean something else in a regex)
# (\d+)    one or more digits, captured
# CITATION.findall("X [1] and Y [3].") → ["1", "3"]
```

The rules map one-to-one to the eval sheet's answer criteria:
**self-contained, source-grounded, faithful, relevant**.

---

## 3. The class and `load()`

```python
@dataclass
class Generator:
    config: Config
    tokenizer: PreTrainedTokenizerBase | None = field(default=None, init=False)
    llm: PreTrainedModel | None = field(default=None, init=False)
    files: dict[str, str] = field(default_factory=dict, init=False)   # file contents cache
```

`Generator(config)` loads **nothing**: `Pipeline.__init__` creates one for
every command, even `search`.

```python
def load(self) -> None:
    if self.tokenizer is None or self.llm is None:     # "or": retry a half-failed load
        now = time.perf_counter()
        print(f"Loading {self.config.llm}...", file=sys.stderr)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.config.llm, dtype=torch.float32)
        except (OSError, ValueError) as e:              # no internet, bad name, disk full
            raise RAGError(f"cannot load model '{self.config.llm}': {e}") from e
        self.llm.eval()
        print(f"Loaded in {time.perf_counter() - now:.1f}s", file=sys.stderr)
```

- **`from_pretrained(name)`** — 1st run: downloads into `~/.cache/huggingface`
  (or `HF_HOME`, set by the Makefile on 42 computers). Then: reads the cache.
- **`AutoTokenizer`** → Qwen's vocabulary (text ↔ token ids) + chat template.
  No `dtype`: a tokenizer has no weights.
- **`AutoModelForCausalLM`** → "Auto" picks the class from the name
  (`Qwen3ForCausalLM`); "CausalLM" = predicts the next token. The `Auto`
  classes are factories, which is why the type hints use the parent classes
  `PreTrainedModel` / `PreTrainedTokenizerBase`.
- **`dtype=torch.float32`** — 4 bytes per weight. Measured on this CPU
  (Ryzen 7 7735HS, no bfloat16 instructions): ~1.7× faster than the default
  `bfloat16`.
- **`eval()`** — inference mode.

---

## 4. `answer()` — the batch loop

```python
def answer(self, search_results: StudentSearchResults) -> StudentSearchResultsAndAnswer:
    if any(r.retrieved_sources for r in search_results.search_results):
        self.load()                         # --k 0 → nothing to answer → Qwen never loaded
    answers: list[MinimalAnswer] = []
    for result in tqdm(search_results.search_results, desc="Answering",
                       file=sys.stderr,     # bar on stderr, not mixed with the output
                       disable=len(search_results.search_results) < 2):   # no bar for 1 question
        answers.append(MinimalAnswer(
            question_id=result.question_id,
            question=result.question,
            retrieved_sources=result.retrieved_sources,   # ALL k sources, not only the prompt ones
            answer=self.generate(result),
        ))
    return StudentSearchResultsAndAnswer(search_results=answers, k=search_results.k)
```

**tqdm** wraps the list: the loop works the same, and each turn redraws the
bar (`Answering: 50%|█████ | 1/2 [00:20<00:20, 20.1s/it]` — done/total,
elapsed, estimated time left, time per question). It redraws in place with
`\r`.

`MinimalAnswer` = `MinimalSearchResults` + `answer` (inheritance), and the
list is still called `search_results`: the answer file is the search file with
an `answer` added to each entry — exactly the subject's example output.

---

## 5. `generate()` — one question

```python
def generate(self, result: MinimalSearchResults) -> str:
    if not result.retrieved_sources:
        return NO_SOURCES
    if self.tokenizer is None or self.llm is None:
        raise RuntimeError("model not loaded")      # never happens; tells mypy they're set
    import torch

    sources = self.select_context(result.retrieved_sources)   # → section 6
    prompt = self.tokenizer.apply_chat_template(
        self.build_messages(result.question, sources),          # → section 7
        tokenize=False,              # return text
        add_generation_prompt=True,  # append "<|im_start|>assistant\n" = your turn
        enable_thinking=False,       # Qwen3: no "<think>...</think>" block
    )
    inputs = self.tokenizer(prompt, return_tensors="pt")
    # {"input_ids": tensor([[151644, 8948, …]]), "attention_mask": …}  shape (1, ~1000)
    # "pt" = PyTorch tensors, what generate() expects
    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():                     # no training bookkeeping: faster
        output = self.llm.generate(
            **inputs,                                # input_ids=…, attention_mask=…
            max_new_tokens=self.config.max_new_tokens,   # 256 max
            do_sample=False)                         # greedy: same input → same answer
    # output = the prompt tokens + the new answer tokens, in one tensor

    new_tokens = output[0][prompt_length:]           # keep only the answer
    answer: str = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return self.add_citations(answer.strip(), sources)       # → section 8
```

Where the time goes (measured): reading a ~1400-token prompt ≈ **9 s**, then
each answer token adds more. Answers are short (30–60 tokens), so **the
prompt size is the main speed knob**.

---

## 6. `select_context()` — the token budget

```python
budget = self.config.max_context_tokens               # 1200
for source in retrieved[:self.config.max_context_sources]:   # best 5 of the k, best first
    ids = self.tokenizer(self.source_text(source))["input_ids"]   # count its tokens
    if not ids:
        continue                                      # unreadable file → skip
    if len(ids) > budget:
        if budget < 50:                               # tiny leftover: not worth it
            break
        ids = ids[:budget]                            # cut to what's left
    selected.append((source, self.tokenizer.decode(ids)))
    budget -= len(ids)
    if budget <= 0:
        break
```

```
source 1: 520 tokens → fits         → budget 1200 → 680
source 2: 510 tokens → fits         → budget  680 → 170
source 3: 480 tokens → cut to 170   → budget    0 → stop
sources 4–10: not read by Qwen (still in the output's retrieved_sources)
```

**k vs what Qwen reads** — `--k 10` retrieves 10 sources and the output lists
all 10. The prompt gets the best ones until the budget is full: that is the
eval sheet's "passes retrieved context to LLM **within token limits**". The
token budget is the real limit (≈2–3 full chunks fit); `max_context_sources`
is just an upper bound.

Measured on 8 questions (fp32):

| Budget | Avg time per answer | Quality |
|---|---|---|
| 2000 tokens | 18.1 s | baseline |
| **1200 tokens** | **13.5 s** | same or better on 7/8 |

---

## 7. `source_text()` and `build_messages()`

```python
def source_text(self, source: MinimalSource) -> str:
    if source.file_path not in self.files:
        try:
            with open(source.file_path, encoding="utf-8", newline="") as f:
                self.files[source.file_path] = f.read()     # read each file once
        except (OSError, UnicodeDecodeError):
            self.files[source.file_path] = ""               # → skipped by select_context
    content = self.files[source.file_path]
    return content[source.first_character_index:source.last_character_index]
```

`answer_dataset`'s input file has only `file_path` + offsets, **no text**.
The text is recovered by slicing the original file — possible because the
chunker's offsets are exact (same `open()` options: `utf-8` + `newline=""`).

```python
@staticmethod
def build_messages(question, sources) -> list[dict[str, str]]:
    context = "\n\n".join(f"[{i}] File: {source.file_path}\n{text}"
                          for i, (source, text) in enumerate(sources, start=1))
    user = f"Sources:\n\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]
```

Qwen is a **chat** model trained on roles (system = how to behave, user = the
request, assistant = its reply). `apply_chat_template` turns this list into
Qwen's exact format (`<|im_start|>system …<|im_end|>` …) — never written by
hand.

---

## 8. `add_citations()` — why it exists

The eval sheet checks that an answer **"cites the source(s) it draws from"**.
Qwen sees numbered sources, so it writes `[1]` — meaningless to the evaluator,
who only sees the answer. `add_citations` translates the numbers into file
paths:

```
...is `/v1/load_lora_adapter`. This endpoint sends a POST request with the adapter details. [1]

Sources:
[1] data/raw/vllm-0.10.1/docs/features/lora.md
```

```python
answer = answer.replace(NOT_IN_SOURCES, "").strip()   # answered AND refused → keep the answer
if not answer:
    return NOT_IN_SOURCES                             # only refused
cited = sorted({int(n) for n in CITATION.findall(answer)
                if 1 <= int(n) <= len(sources)})      # valid numbers, no duplicates
numbers = cited or list(range(1, len(sources) + 1))   # cited nothing → list every source given
files: dict[str, list[int]] = {}
for n in numbers:
    files.setdefault(sources[n - 1][0].file_path, []).append(n)   # [1] → index 0
lines = ["".join(f"[{n}]" for n in ns) + f" {path}" for path, ns in files.items()]
return answer + "\n\nSources:\n" + "\n".join(lines)  # one line per file: "[1][2] …/sampler.py"
```

Why not ask Qwen to write the paths itself: a 0.6B model copies long paths
badly (hallucinated folders), paths cost ~20 tokens each vs 3 for `[1]`, and
it sometimes cites nothing at all. Qwen writes short markers; the code, which
built the prompt, writes the exact paths.

---

## 9. Speed experiments (Ryzen 7 7735HS, CPU only)

| Option | Result | Kept? |
|---|---|---|
| float32 instead of bfloat16 | ~1.7× faster, same answers | ✅ |
| context budget 2000 → 1200 tokens | 18.1 s → 13.5 s, same quality | ✅ |
| 16 threads instead of 8 | slower (34 s vs 24 s) | ❌ default 8 = physical cores |
| several questions batched in one `generate` | slower (88 s vs 58 s for 4) | ❌ |
| int8 dynamic quantization | broken answers (`</think>`, pasted code, wrong refusals) | ❌ |
| `max_new_tokens` 150 | no change: answers are shorter anyway | ➖ |

Loading the model costs ~6 s plus 3–7 s of imports **per command**; the HTTP
server would pay it once.

---

## 10. Errors

| Situation | What happens |
|---|---|
| no sources (k=0, gibberish) | `NO_SOURCES`, model never loaded |
| model can't be downloaded/loaded | `RAGError` → "Error: cannot load model ..." exit 1 |
| a source file can't be read | that source is skipped |
| sources don't contain the answer | `NOT_IN_SOURCES` (the model is told to say so) |
| answers but also adds the refusal | refusal removed, answer kept |
