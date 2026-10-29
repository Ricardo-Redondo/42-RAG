*This project has been created as part of the 42 curriculum by rsao-pay.*

# RAG against the machine

## Description

A **Retrieval-Augmented Generation** (RAG) system that answers questions about
the [vLLM](https://github.com/vllm-project/vllm) codebase (version 0.10.1).

A language model only knows what it was trained on. Instead of retraining it,
RAG looks up the relevant information at question time and hands it to the
model:

1. **Indexing** — every Python, Markdown and text file of the repository is
   cut into chunks of at most 2000 characters, and a BM25 index is built over
   them.
2. **Retrieval** — for a question, BM25 returns the top-k chunks, each as a
   source location: `file_path` + character range.
3. **Augmenting** — the best sources are placed in the model's prompt, within
   a token budget.
4. **Generating** — `Qwen/Qwen3-0.6B` writes a short answer grounded in those
   sources, citing them.

Retrieval is measured with **recall@k**: the share of the true sources found
in the first k results (same file, character ranges overlapping with an IoU of
at least 0.05).

---

## Instructions

### Requirements

- Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/)
- A few GB of free disk space (PyTorch CPU + the Qwen weights, downloaded on
  the first `answer`). On 42 computers the Makefile stores the venv, the uv
  cache and the Hugging Face cache in `/sgoinfre` automatically.
- The corpus in `data/raw/vllm-0.10.1/` and the datasets in `data/datasets/`
  (not part of the repository).

### Installation

```sh
make install        # = uv sync (CPU-only PyTorch)
```

### Running

Every command is `uv run python -m src <command> [options]`:

| Command | What it does |
|---|---|
| `index --max_chunk_size 2000` | chunk `data/raw/` and save the index under `data/processed/` |
| `search "<question>" --k 10` | print the top-k sources for one question |
| `search_dataset --dataset_path <file> --k 10 --save_directory <dir>` | search a whole dataset, write a `StudentSearchResults` JSON |
| `answer "<question>" --k 10` | retrieve + generate an answer for one question |
| `answer_dataset --student_search_results_path <file> --save_directory <dir>` | generate answers for saved search results |
| `evaluate --student_search_results_path <file> --dataset_path <file>` | our own recall@1/3/5/10 against a ground-truth dataset |

Global options (any command): `--raw_path`, `--processed_path`, `--llm`.

### Make shortcuts

| Rule | Runs |
|---|---|
| `make install`, `run`, `debug`, `lint`, `lint-strict` | subject rules |
| `make clean` / `fclean` / `vclean` | caches / + index and outputs / + the venv |
| `make index`, `search`, `answer` | one command (`QUERY="..."`, `K=10`, `CHUNK=2000`) |
| `make search_dataset`, `answer_dataset`, `evaluate`, `moulinette` | one dataset (`TYPE=docs\|code`, `SCOPE=public\|private`) |
| `make check` | index → search docs + code → recall (like the exam) |
| `make edge_cases` | the 4 degenerate inputs of the exam |

---

## Example usage

```sh
$ uv run python -m src index --max_chunk_size 2000
Ingestion complete! 14547 chunks from 1969 files in 4.5s. Indices saved under data/processed/

$ uv run python -m src search "How to configure OpenAI server?" --k 3
data/raw/vllm-0.10.1/examples/online_serving/openai_transcription_client.py [0/712]
data/raw/vllm-0.10.1/examples/online_serving/openai_chat_completion_client_with_tools_required.py [0/565]
data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [0/1952]

$ uv run python -m src answer "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --k 10
The HTTP endpoint used to dynamically load a LoRA adapter in vLLM is `/v1/load_lora_adapter`.
This endpoint sends a POST request with the adapter details. [1]

Sources:
[1] data/raw/vllm-0.10.1/docs/features/lora.md

Retrieved sources:
  data/raw/vllm-0.10.1/docs/features/lora.md [4694/6099]
  data/raw/vllm-0.10.1/docs/features/lora.md [4074/4694]
  data/raw/vllm-0.10.1/vllm/entrypoints/openai/serving_models.py [10620/11282]
  ... (10 in total)

$ uv run python -m src search_dataset \
    --dataset_path data/datasets/public/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json

$ uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/public/AnsweredQuestions/dataset_docs_public.json
Evaluation Results
========================================
Recall@1: 0.630  Recall@3: 0.810  Recall@5: 0.830  Recall@10: 0.880

$ uv run python -m src search "" --k 10
No relevant sources found.

$ uv run python -m src search "lora" --k abc
Error: --k must be an integer >= 0, got 'abc'
```

---

## System architecture

```
terminal ──▶ Cli (cli.py) ──▶ Pipeline (pipeline.py) ──┬──▶ Indexer    (indexing/)
             checks args       reads/writes JSON,       ├──▶ Retriever  (retrieval/)
                               raises RAGError          ├──▶ Generator  (generation/)
                                                        └──▶ recall_at_k (evaluation/)
__main__.py: runs Fire(Cli), turns every error into one line + exit code (never a traceback)
```

| Component | Role |
|---|---|
| `src/cli.py` | Python Fire CLI: one method per command. Validates `k`, wraps the question, prints. |
| `src/pipeline.py` | Connects the stages, reads/validates input JSON, writes output JSON. Also the entry point for a future HTTP server. |
| `src/indexing/` | Two chunking strategies (Python / Markdown) + BM25 indexing and saving. |
| `src/retrieval/` | Lazy-loads the index, searches a batch of questions, returns `StudentSearchResults`. |
| `src/generation/` | Lazy-loads Qwen, builds the prompt within a token budget, generates and adds citations. |
| `src/evaluation/` | recall@k — gives exactly the moulinette's numbers. |
| `src/text.py` | `expand_identifiers`, shared by indexing and retrieval. |
| `src/models.py`, `src/config.py`, `src/errors.py` | pydantic data models, settings, user-facing error type. |

**Each command is a separate process**: `index` saves the index to disk,
`search` loads it back. Everything is **lazy**: `search` never imports
`transformers`/`torch` (3–7 s) nor loads Qwen; `answer --k 0` answers
without loading the model.

Every folder of `src/` has its own `README.md` with the details and the code
walked through.

---

## Chunking strategy

Two distinct strategies, both from LangChain's `RecursiveCharacterTextSplitter`
(`chunk_size=2000`, `chunk_overlap=0`):

- **Python files** (`Language.PYTHON`) — cuts preferably before a top-level
  `class` or `def`, then at blank lines, then lines, then words.
- **Markdown / text files** (`Language.MARKDOWN`) — cuts preferably before a
  heading (`#` … `######`), then code fences and rules, then blank lines,
  lines, words.

"Recursive": a separator is only used if a piece is still longer than the
chunk size; small neighbouring pieces are merged back up to the limit.

**Exact offsets.** The grader compares character ranges, so every chunk must
satisfy `file[start:end] == chunk`. Files are read in text mode with
`encoding="utf-8"` (offsets count characters, not bytes) and `newline=""`
(`\r\n` stays 2 characters), and the splitter runs with
`strip_whitespace=False` (the default removes characters at chunk edges).
Checked on all 1969 files: 0 offset errors, no character lost, no chunk over
2000 characters.

Only `.py`, `.md` and `.txt` are indexed: these are the only extensions that
appear in the ground-truth sources (200 / 194 / 5).

**Chunk size.** Too **small**: a chunk holds too few words to match a
question and loses its context (a function body without its signature);
there are many more chunks, and the top-5 covers less text. Too **large**:
relevant lines are diluted among unrelated text, and the prompt for the model
gets longer and slower. Here the grader's low overlap bar (IoU ≥ 0.05)
favours large chunks, and measured recall is almost flat — see the
performance analysis.

---

## Retrieval method

**BM25** (via `bm25s`, `k1 = 1.5`, `b = 0.75`, Lucene variant). For each word
of the question found in a chunk, BM25 adds a score that:

- grows with how **rare** the word is across all chunks (IDF): `pagedattention`
  is worth much more than `model`;
- grows with how often the word appears in the chunk, but **saturates** (`k1`):
  5 occurrences are not worth 5×1;
- is **normalized by chunk length** (`b`): long chunks contain more words by
  chance.

The chunk's score is the sum over the question's words; chunks are ranked by
score. **TF-IDF** is the older form of the same idea — term frequency ×
inverse document frequency — without the saturation and the length
normalization.

Text processing, identical for chunks and questions:

1. **Identifier splitting** (`src/text.py`): `max_model_len` → also
   `max model len`, `TritonAttentionMetadataBuilder` → also
   `Triton Attention Metadata Builder` (the original word is kept). Chunks also
   get the words of their **file path** (`vllm/lora/worker_manager.py` →
   `vllm lora worker manager`).
2. Lowercase, split into words, remove **English stopwords**.
3. **Stemming** (PyStemmer, English): `load / loaded / loading` → `load`.

Results with a score of 0 (no word in common: empty or gibberish questions)
are dropped. `k` is capped at the number of chunks. A whole dataset is
searched in one batch: 200 questions in ~0.7 s.

---

## Answer generation

`Qwen/Qwen3-0.6B` (default; another Hugging Face causal LM can be passed with
`--llm`, but everything is tested with Qwen3-0.6B), on CPU in float32.

- **Context within a token budget**: the best retrieved sources (at most 5)
  are added until 1200 tokens are used, the last one cut to fit. The output
  still lists all k retrieved sources.
- **Prompt**: a system message with 5 rules (answer directly in 2–4
  self-contained sentences; cite sources as `[n]`; only what the sources say;
  don't describe the sources; an exact refusal sentence when the sources don't
  answer), then the numbered sources with their file paths, then the question.
  Qwen3's "thinking" mode is turned off.
- **Greedy decoding** (`do_sample=False`): the same question always gets the
  same answer.
- **Citations**: the `[n]` markers are translated into a `Sources:` list of
  file paths after generation, so every answer names real files even when the
  small model forgets to cite.

---

## Performance analysis

### Retrieval (recall@5, k = 10, chunk size 2000)

| | docs public | code public | docs private | code private |
|---|---|---|---|---|
| **final system** | **83.0%** | **82.8%** | **82.0%** | **75.0%** |
| required | 80% | 50% | 80% | 50% |

Private scores confirmed with the moulinette. Exam timings: indexing ~4.5 s
(limit 300 s), 200 questions ~0.7 s (limit 90 s).

### What each change did (recall@5, public, docs / code)

| Version | docs | code |
|---|---|---|
| fixed 2000-character windows, plain BM25 tokens | 85.0 | 59.6 |
| LangChain Python/Markdown splitters, plain tokens | 82.0 | 53.5 |
| + identifier splitting + file path words (final) | 83.0 | **82.8** |

Other chunkers tried (same final tokenization): fixed windows 84.0 / 85.9,
fixed windows cut at the nearest `def`/heading 81.0 / 85.9, overlapping
windows 85.0 / 85.9. Chonkie's code chunker was rejected: 588 wrong offsets
and chunks over 2000 characters.

### Effect of the chunk size (final system, public)

| `--max_chunk_size` | chunks | docs R@1 | docs R@5 | code R@1 | code R@5 |
|---|---|---|---|---|---|
| 500 | 59,816 | 0.620 | 0.820 | 0.545 | 0.828 |
| 1000 | 29,523 | 0.690 | 0.830 | 0.515 | 0.808 |
| 1500 | 19,407 | 0.680 | 0.830 | 0.535 | 0.798 |
| 2000 | 14,547 | 0.630 | 0.830 | 0.525 | 0.828 |

Once identifiers are split, recall@5 is almost flat (80–83%) across sizes;
mid-size chunks rank the right docs chunk first slightly more often (R@1).
2000 is kept: 4× fewer chunks than 500, and every chunk carries more context
for the model.

### Generation (CPU: AMD Ryzen 7 7735HS)

| Setting | Average time per answer |
|---|---|
| bfloat16, 2000-token context | ~47 s |
| float32, 2000-token context | ~18 s |
| **float32, 1200-token context (final)** | **~13.5 s** |

Plus ~6 s to load the model and 3–7 s of imports, once per command. Rejected
after measuring: 16 threads (slower than the default 8), batching several
questions in one `generate` call (slower on CPU), int8 quantization (broken
answers).

---

## Design decisions

- **BM25 over TF-IDF**: saturation and length normalization handle chunks of
  different sizes better; `bm25s` builds the index in seconds and searches a
  whole dataset in one batched call.
- **Tokenization over chunking**: the biggest gain (+29 points on code) came
  from splitting identifiers, not from smarter chunk boundaries — questions
  are written in English ("max model length"), code in `snake_case`.
- **LangChain splitters** for two real strategies (required) with exact
  offsets; same recall as plain fixed windows once tokenization is fixed.
- **The chunk text lives in the index** (`bm25s` `corpus=`), so search returns
  the chunk directly. `answer_dataset` gets the text back from
  `file_path` + offsets, which works because the offsets are exact.
- **Lazy loading and lazy imports**: every command only pays for what it uses.
- **One error path**: the pipeline raises `RAGError` with a user-facing
  message; `__main__` prints it and exits with code 1. No input can produce a
  traceback.
- **pydantic everywhere data crosses a stage**: the subject's models, the
  config (validated on assignment, unknown fields forbidden) and the chunks.
- **Context budget of 1200 tokens** and **float32**: chosen by measuring
  speed and answer quality, not guessed.
- **Citations added in code** instead of asking a 0.6B model to copy file
  paths: shorter output, no hallucinated paths.

---

## Challenges faced

- **Exact character offsets.** Bytes vs characters (`é` is 2 bytes), `\r\n`
  translation by `open()`, and splitters that strip whitespace all shift the
  positions. Solved by reading text with `newline=""`, `strip_whitespace=False`,
  and a check over the whole corpus that `file[start:end] == chunk`.
- **Code recall.** Code questions scored ~55% because BM25 sees
  `max_model_len` as one unknown word. Identifier splitting raised it to ~80%.
- **Degenerate inputs.** BM25 returns k random chunks with score 0 for an
  empty or gibberish query, and raises if k is larger than the corpus. Handled
  by filtering score-0 results and capping k.
- **Slow generation on CPU.** Profiling showed reading the prompt dominates
  (~9 s for 1400 tokens). Solved with float32 and a smaller context budget;
  quantization was tried and rejected because it broke the answers.
- **A small model's habits.** Qwen3-0.6B sometimes answers *and* adds the
  refusal sentence, rambles about the sources, or forgets to cite. Handled
  with stricter rules in the prompt and post-processing.
- **Startup time.** Importing `transformers` takes 3–7 s; importing it lazily
  keeps `search` fast.

---

## Resources

- vLLM documentation — https://docs.vllm.ai
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* (2009) — the BM25 formula
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval*,
  chapters 6 (TF-IDF) and 8 (evaluation, recall) —
  https://nlp.stanford.edu/IR-book/
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (2020) — https://arxiv.org/abs/2005.11401
- `bm25s` — https://github.com/xhluca/bm25s
- LangChain text splitters —
  https://python.langchain.com/docs/concepts/text_splitters/
- Qwen3-0.6B model card — https://huggingface.co/Qwen/Qwen3-0.6B
- Hugging Face Transformers documentation —
  https://huggingface.co/docs/transformers
- Python Fire — https://github.com/google/python-fire
- pydantic — https://docs.pydantic.dev

### Use of AI

An AI assistant was used during this project:

- **Understanding**: understanding of unkown librarys: `bm25s`, `transformers`,
  `PyTorch`, `Fire`.
- **Experiments**: measuring chunkers, tokenization, chunk sizes and
  generation settings (the tables in this README).
- **Documentation**: reviewing of Readme's for typos and badly written content.
