# `retrieval/` — from a question to the top-k sources

This folder answers: *"which chunks best match this question?"* It runs in
the `search` and `search_dataset` commands, and as the first step of `answer`.

```
Pipeline.search / search_dataset / answer
   │  wraps the question(s) in a RagDataset
   ▼
Retriever.retrieve(querys, k)
   │  1. load()                    BM25.load(...) only the first time (lazy)
   │  2. cap k                     k=0 → empty results, k > chunks → all chunks
   │  3. tokenize(questions)       same settings as the chunks!
   │  4. bm25.retrieve(tokens, k)  → results + scores
   │  5. pair each question with its row, drop score-0 results
   │  6. build MinimalSearchResults → StudentSearchResults
   ▼
StudentSearchResults  ──▶ printed (search), saved as JSON (search_dataset)
                          or passed to the generator (answer)
```

---

## 1. Loading the index: `load()`

The `index` command ran in another process and is gone. Everything comes back
from the files it saved in `<processed_path>/bm25_index/`.

```python
retriever: BM25 | None = field(default=None, init=False)   # starts empty

def load(self) -> BM25:
    if self.retriever is None:                              # first call only
        path = Path(self.config.bm25_index_path)
        if not (path / "params.index.json").is_file():
            raise RAGError("no index found in '...', run the index command first")
        try:
            self.retriever = BM25.load(str(path), load_corpus=True, mmap=True,
                                       show_progress=False)
        except (OSError, ValueError) as e:
            raise RAGError("the index in '...' is unreadable ...") from e
    return self.retriever
```

- **`field(default=None, init=False)`** — starts as `None` and is **not** a
  constructor argument: `Retriever(config, stemmer)` is the only way to create
  it, nobody can pass a BM25 in by mistake.
- **Why lazy**: not in `__init__`, because the `Retriever` is created for
  *every* command, including `index` when no index exists yet; not on every
  call, because `search_dataset` would reload it for nothing.
- **`params.index.json` check** — every saved index has this file. Checking it
  first gives a clear "run the index command first" instead of a bm25s error.
- **Returns the BM25** — so `retrieve` writes `bm25 = self.load()` and mypy
  knows `bm25` is a `BM25`, never `None`.

**`BM25.load(folder, load_corpus=True, mmap=True)`**

- It is a **classmethod**: you call it on the class (`BM25.load(...)`) and it
  **creates and returns a new `BM25` object** filled from the files.
  Calling it on an existing object without assigning the result does nothing
  useful: the new object is thrown away.
- `load_corpus=True` → your chunks are loaded into the attribute `.corpus`.
  Without it, `.corpus` is `None`.
- `mmap=True` → the arrays are read from disk *when needed* instead of all at
  once → faster start. With mmap, `.corpus` is a `JsonlCorpus` (a lazy reader)
  instead of a plain `list`, but you use it the same way.
- `show_progress=False` → no progress bar for something this short.
- Nothing is recomputed: all BM25 scores were computed by `index`. Loading is
  just reading files.

---

## 2. Capping `k`

```python
top_k = min(k, bm25.scores["num_docs"])
if top_k <= 0 or not questions:
    return ... every question with retrieved_sources=[] ...
```

- `bm25.scores["num_docs"]` is the number of chunks in the index (14547).
- **`k` larger than the number of chunks** → bm25s raises `ValueError`
  (tested), so `k` is capped.
- **`k = 0`** → no search at all, every question gets an empty list (that's
  the `answer "What is vLLM?" --k 0` edge case).
- The output still says `k=k` (what was asked), not `top_k`.

---

## 3. Tokenizing the questions

```python
many = len(questions) > 1
querys_tokens = tokenize([expand_identifiers(q.question) for q in questions],
                         stopwords="en", stemmer=self.stemmer,
                         show_progress=many)
```

Same function as in indexing, and it **must use the same settings**:

- **`expand_identifiers`** (from `src/text.py`, see `indexing/README.md`):
  the chunks were indexed with their snake_case/CamelCase names split
  (`max_model_len` → `max model len`), so the question goes through the same
  function — a question quoting `max_model_len` also matches chunks that only
  say "max model". Both sides import the **one** function, so they can never
  drift apart.

- **Stemmer: required.** The index stores stemmed words (`loaded` → `load`).
  An unstemmed question looks up `loaded`, which doesn't exist → no match.
  That's why the indexer and the retriever share **one** `Stemmer`, created in
  `Pipeline.__init__`.
- **Stopwords:** a stopword left in the question is simply not in the
  vocabulary → ignored. Not harmful, but both sides are kept identical anyway.
- **`show_progress=many`** — progress bars only for a whole dataset, not for
  one question.

Passing a **list** of questions tokenizes them all at once.

---

## 4. Searching: `bm25.retrieve`

```python
results, scores = bm25.retrieve(querys_tokens, bm25.corpus, top_k, show_progress=many)
```

(signature: `retrieve(query_tokens, corpus=None, k=10, ...)`)

### What it returns — the one thing to remember

**One object that holds two arrays.** Unpack it into two variables.

```python
results   # numpy array, shape (number_of_questions, k)
scores    # numpy array, same shape, float32
```

`results[i][j]` and `scores[i][j]` = the j-th best chunk for question i. Rows
are in the **same order as the questions** you tokenized. Inside each row,
best first.

With `load_corpus=True`, each cell of `results` is **your chunk dict**,
exactly as saved:

```python
results = [                                   # 2 questions, k = 2
    [ {"chunk": "...", "source": {...}}, {"chunk": "...", "source": {...}} ],  # question 0
    [ {"chunk": "...", "source": {...}}, {"chunk": "...", "source": {...}} ],  # question 1
]
scores = [[9.81, 7.42],
          [5.10, 4.98]]
```

(Without a loaded corpus, the cells would be chunk **numbers** instead.)

### Types of what comes back

| Thing | Type |
|---|---|
| what `retrieve` returns | a small named tuple (`Results`) → unpack it |
| `results` | `numpy.ndarray`, dtype `object` (holds dicts) |
| `results[i][j]` | `dict` — one chunk |
| `results[i][j]["source"]` | `dict` — a dumped `MinimalSource` |
| `scores` | `numpy.ndarray`, dtype `float32` |

**`.tolist()`** turns a numpy array into plain Python lists with the same
nesting, and converts numpy numbers into Python ones (`float32` → `float`,
`int64` → `int`). For dicts in an object array, numpy gives **no speed
advantage** — it's only fast for numbers with vectorized operations.

bm25s has no type hints, so mypy sees its return values as `Any`. They are
converted to our own pydantic models right away so the rest of the code is
typed.

---

## 5. Building the output

```python
for query, row, row_scores in zip(questions, results.tolist(), scores.tolist()):
    search_results.append(MinimalSearchResults(
        question_id=query.question_id,
        question=query.question,
        retrieved_sources=[item["source"]
                           for item, score in zip(row, row_scores) if score > 0],
    ))
return StudentSearchResults(search_results=search_results, k=k)
```

### `zip`: walking several lists side by side

`zip(questions, rows, score_rows)` gives `(question 0, row 0, scores 0)`,
`(question 1, row 1, scores 1)`, ... This works because `retrieve` keeps the
rows in the questions' order. (A dict keyed by question doesn't work: pydantic
models and lists are **unhashable** → `TypeError`.)

The inner `zip(row, row_scores)` pairs each chunk with its score.

### Dropping score-0 results

A score of **0** means *no word of the question appears in the chunk*. That
happens for an **empty** question or **gibberish** (`asdfghjkl`): bm25s still
returns k chunks, all with score 0 — random files. Keeping only `score > 0`
turns that into "no sources", which is what the edge cases expect. Real
questions always share words with their top chunks, so recall is unchanged
(checked: same numbers as before the filter).

### `item["source"]` → `MinimalSource`

`item["source"]` is a **dict** (`file_path`, `first_character_index`,
`last_character_index`) — the key is `"source"` (singular, the field name in
`Chunk`). **pydantic converts the dicts for you**: `retrieved_sources` is
typed `List[MinimalSource]`, so pydantic checks each dict and builds a
`MinimalSource` from it. A bad dict → `ValidationError`.

### One method for every command

`Retriever.retrieve` takes a **`RagDataset`** (a list of questions). For a
single question the pipeline wraps it:
`retrieve(RagDataset(rag_questions=[question]), k).search_results[0]`.
One code path, and `search_dataset` gets batching (one tokenize + one retrieve
for all questions) for free — 200 questions take ~1 s, far under the 90 s
limit.

The CLI receives the question as text and builds
`UnansweredQuestion(question=str(query))` — `question_id` gets a random uuid
automatically (`default_factory`).

---

## 6. Errors

| Situation | What happens |
|---|---|
| no index (never ran `index`, or wrong `--processed_path`) | `RAGError`: "no index found in '...', run the index command first" |
| index files broken | `RAGError`: "the index in '...' is unreadable, run the index command again" |
| `k = 0` | empty sources, no error |
| `k` > number of chunks | capped, no error |
| empty / gibberish question | empty sources ("No relevant sources found.") |
