# `retrieval/` — from a question to the top-k sources

This folder answers: *"which chunks best match this question?"* It runs in
the `search` and `search_dataset` commands (and later inside `answer`).

```
Pipeline.search / search_dataset
   │  wraps the question(s) in a list
   ▼
Retriever.retrieve(questions, k)
   │  1. BM25.load(...)            (only the first time — lazy loading)
   │  2. tokenize(questions)       (same settings as the chunks)
   │  3. bm25.retrieve(tokens, k)  → results + scores
   │  4. pair each question with its row of results
   │  5. build MinimalSearchResults → StudentSearchResults
   ▼
StudentSearchResults  ──▶ printed (search) or saved as JSON (search_dataset)
```

---

## 1. Loading the index

The `index` command ran in another process and is gone. Everything comes back
from the files it saved in `data/processed/bm25_index/`.

**`BM25.load(folder, load_corpus=True, mmap=True)`**

- It is a **classmethod**: you call it on the class (`BM25.load(...)`) and it
  **creates and returns a new `BM25` object** filled from the files.
  Calling it on an existing object (`self.retriever.load(...)`) without
  assigning the result does nothing useful: the new object is thrown away.
- It always returns a `BM25` object. The options change what's inside it:
  - `load_corpus=True` → your chunks are loaded into the attribute
    `.corpus`. Without it, `.corpus` is `None`.
  - `mmap=True` → the arrays are read from disk *when needed* instead of all
    at once → faster start. With mmap, `.corpus` is a `JsonlCorpus` (a lazy
    reader) instead of a plain `list`, but you use it the same way.
- Nothing is recomputed: all BM25 scores were computed by `index`. Loading is
  just reading files.
- If the folder doesn't exist (someone runs `search` before `index`), it
  raises an error → must be caught with a clear message, no traceback.

### Lazy loading (what `retriever: BM25 | None = None` is for)

```python
retriever: BM25 | None = None       # starts empty

if self.retriever is None:          # first call only
    self.retriever = BM25.load(...)
```

- **Not in `__init__`**: the `Retriever` is created for *every* command,
  including `index`, when no index exists yet.
- **Not on every call**: `search_dataset` would reload it for nothing.
- After the `if`, mypy knows `self.retriever` is a `BM25`, not `None`
  ("type narrowing").

---

## 2. Tokenizing the questions

**`tokenize([q.question for q in questions], stopwords="en", stemmer=self.stemmer)`**

Same function as in indexing, and it **must use the same settings**:

- **Stemmer: required.** The index stores stemmed words (`loaded` → `load`).
  An unstemmed question looks up `loaded`, which doesn't exist → no match.
- **Stopwords:** a stopword left in the question is simply not in the
  vocabulary → ignored (score 0). Not harmful, but keep both sides identical
  anyway.

Passing a **list** of questions tokenizes them all at once.

---

## 3. Searching: `bm25.retrieve`

**`results, scores = self.retriever.retrieve(tokens, corpus, k)`**

(signature: `retrieve(query_tokens, corpus=None, k=10, ...)` — writing `k=k`
by name is clearer.)

### What it returns — the one thing to remember

**One object that holds two arrays.** Unpack it into two variables.

```python
results   # numpy array, shape (number_of_questions, k)
scores    # numpy array, same shape, float32
```

`results[i][j]` and `scores[i][j]` = the j-th best chunk for question i. Rows
are in the **same order as the questions** you tokenized. Inside each row,
best first.

With `load_corpus=True` , each cell of `results` is **your chunk
dict**, exactly as saved:

```python
results = [                                   # 2 questions, k = 2
    [ {"chunk": "...", "source": {...}}, {"chunk": "...", "source": {...}} ],  # question 0
    [ {"chunk": "...", "source": {...}}, {"chunk": "...", "source": {...}} ],  # question 1
]
scores = [[9.81, 7.42],
          [5.10, 4.98]]
```

(Without a loaded corpus, the cells would be chunk **numbers** instead)

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
`int64` → `int`). Useful because:
- `json.dumps` can't serialize `numpy.float32`;
- plain lists are easier to annotate: `list[list[dict[str, Any]]]`.

For dicts in an object array, numpy gives **no speed advantage** — it's only
fast for numbers with vectorized operations (`scores > 0`).

bm25s has no type hints, so mypy sees its return values as `Any`. Converting to
our own types right away so the rest of the code is typed.

---

## 4. Building the output

### One question + one row → one `MinimalSearchResults`

```
question_id        ← question.question_id
question           ← question.question
retrieved_sources  ← [item["source"] for item in row]
```

`item["source"]` is a **dict** (`file_path`, `first_character_index`,
`last_character_index`) — not a string, and not yet a `MinimalSource`.
The key is `"source"` (singular, the field name in `Chunk`).

**pydantic converts the dicts for you**: `retrieved_sources` is typed
`List[MinimalSource]`, so when you pass a list of dicts pydantic checks each
one and builds a `MinimalSource` from it. A bad dict → `ValidationError`.

### Many questions → pair them with `zip`

**`zip(questions, rows)`** walks two lists side by side:
`(question 0, row 0)`, `(question 1, row 1)`, ... This works because
`retrieve` keeps the rows in the questions' order. (A dict keyed by question
doesn't work: pydantic models and lists are **unhashable** → `TypeError`.)

### Wrapping up: `StudentSearchResults`

```python
StudentSearchResults(search_results=[...], k=k)
```

Both fields **by name**; `k` is required. To save it:
`result.model_dump_json(indent=...)` → a JSON string to write to a file.

### One method for both commands

`Retriever.retrieve` takes a **list** of questions. For a single question the
pipeline wraps it: `retrieve([question], k).search_results[0]`. One code path,
and `search_dataset` gets batching (one tokenize + one retrieve for all
questions) for free — important for the 90-second limit.

The CLI `search` receives a **string**: build the question with
`UnansweredQuestion(question=text)` — `question_id` gets a random uuid
automatically (`default_factory`).
