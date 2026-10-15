# `indexing/` — from files to a searchable index

This folder does two jobs, run by the `index` command:

1. **Chunking** (`chunking.py`) — cut every file into pieces ("chunks") and
   remember exactly where each piece comes from.
2. **Indexing** (`indexing.py`) — turn the chunks into a BM25 index and save it
   to disk so `search` can use it later.

```
Pipeline.index()
   │  finds every file, picks a chunker for each (by extension)
   ▼
Indexer.index(strats)
   │  1. chunker.proccess(path, size)  → list[Chunk]   (for every file)
   │  2. tokenize(texts)               → tokens
   │  3. BM25().index(tokens)          → word statistics
   │  4. save(dir, corpus=chunks)      → files on disk
   ▼
data/processed/bm25_index/
```

---

## 1. Chunking

### Why chunk at all

Search never returns a whole file — it returns **chunks**. A chunk is the
smallest thing your system can ever find, and each chunk becomes one possible
`MinimalSource` in the output. The grader checks:

- the **file path** is exactly right, and
- your character range **overlaps** the real answer's range
  (IoU ≥ 0.05 — a low bar).

Hard limit: no chunk may be longer than **2000 characters**, or the
moulinette rejects the whole output file.

### What a chunk is: the `Chunk` model

```python
class Chunk(BaseModel):
    chunk: str              # the text itself (needed by BM25 and later by Qwen)
    source: MinimalSource   # where it comes from: file_path + start + end
```

`MinimalSource` alone has no text, so `Chunk` wraps it with the text.

**The rule that must always hold:**

```python
file_content[source.first_character_index : source.last_character_index] == chunk
```

If this is false for any chunk, your offsets are wrong.

### Offsets: characters, not bytes, and untouched line endings

An offset is a position number: "character 4695". Two things can make your
positions differ from the grader's:

| Problem | Example | Fix |
|---|---|---|
| Bytes vs characters | `é` is 1 character but 2 bytes in UTF-8, so byte positions drift after it | Read in **text** mode: a Python `str` counts characters |
| Line-ending translation | by default `open()` turns `\r\n` (2 chars) into `\n` (1 char) | `newline=""` keeps line endings as they are on disk |

In the corpus: 120 `.py`/`.md` files contain non-ASCII characters, 4 have
`\r\n` line endings. (Small effect on the score, but your offsets should be
exact.)

### The chunker classes

```python
class Chunking(ABC):              # the "shape" every chunker must have
    def proccess(path, chunking_size) -> list[Chunk]

class PyChunking(Chunking): ...   # for .py files
class MdChunking(Chunking): ...   # for .md / .txt files
```

- **`ABC` + `@abstractmethod`** — `Chunking` is a template: it can't be used
  directly, and every subclass *must* implement `proccess`. mypy checks this.
- **`@staticmethod`** — a plain function that lives in the class for
  organisation. It receives **nothing automatically** (no `self`, no `cls`),
  so its parameters are exactly what you pass: `proccess(path, chunking_size)`.
  You call it on the class, `PyChunking.proccess(...)`, without creating a
  `PyChunking()`. That is why the pipeline can store the **classes
  themselves** in a dict.
- **Decorator order** — decorators apply bottom-up, and `abstractmethod` must
  receive the plain function first:

  ```python
  @staticmethod      # on top
  @abstractmethod    # directly above the def
  def proccess(path: Path, chunking_size: int) -> list[Chunk]: ...
  ```

  The other order crashes on import (`'__isabstractmethod__' ... not
  writable`).
- **`@classmethod` vs `@staticmethod`** — a classmethod receives the class as
  its first parameter (`cls`); use it only when the method needs the class
  (e.g. to call another class helper as `cls._helper()`). `proccess` doesn't,
  so static fits.

### Why two strategies (the subject requires both)

A good chunk is **one coherent idea**, and "an idea" looks different per file
type:

- **Markdown**: structure comes from headings (`#`, `##`), then paragraphs
  (blank lines). A section is usually about one topic.
- **Python**: structure comes from functions and classes. The `ast` module can
  give you each `def`/`class` with its start and end line.

### _temporary:_
- _Current state: both classes use the same **fixed-window** logic (a baseline).
That's fine for a first end-to-end run; smarter strategies come after you
have a recall number to beat._

---

## 2. Finding the files (in `pipeline.py`)

**`Path(root).rglob("*")`** — walks `root` and **all subfolders**, yielding
every entry as a `Path`. If `root` is relative (`data/raw/vllm-0.10.1`), every
result starts with that same prefix — exactly the `file_path` format the
grader wants.

**`path.is_file()`** — `True` for files, `False` for folders.

**`path.suffix`** — the extension with the dot: `".py"`, `".md"`, `""` for
`Makefile`. Case-sensitive (`.MD` ≠ `.md`).

Choosing which extensions to index: check which ones appear in the
`file_path`s of the AnsweredQuestions datasets. Indexing files that are never
answers only adds noise and time.

---

## 3. Tokenizing

BM25 works on **words**, not text. `tokenize` turns each chunk's text into a
list of words.

**`bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)`**

- `texts`: one string, or a list of strings (one per chunk).
- What it does, in order:
  1. **lowercase** (`LoRA` → `lora`), because `lower=True` by default;
  2. **split into words** with the regex `token_pattern` (default
     `\w\w+` = 2 or more letters/digits/`_`). Note: `max_model_len` stays ONE
     word because `_` counts as a letter;
  3. **remove stopwords** — very common words with no meaning (`the`, `is`,
     `how`, `to`...). `"en"` = bm25s's built-in English list;
  4. **stem** each word with the stemmer, if one is given.
- Returns a `Tokenized` object: the words as numbers + the vocabulary
  (word → number).
- `show_progress=False` hides its progress bar.

**Stemmer** — `Stemmer.Stemmer("en")` (package `PyStemmer`, imported as
`Stemmer`). Reduces words to a common root so different forms match:
`load, loads, loaded, loading → load`. Roots can look odd (`configur`); that's
fine, they're only used for matching. "Py" in PyStemmer means *written for
Python*, not *for Python code*. It has no type stubs → add `"Stemmer"` to the
mypy `ignore_missing_imports` list.

**The rule:** the **exact same** tokenization (stopwords, stemmer, any custom
splitting) must be used for the chunks **and** for the questions. If the index
contains `load` but the question is tokenized to `loaded`, they never match.
That's why the indexer and the retriever share one `Stemmer` (created in
`Pipeline.__init__`).

---

## 4. BM25 — what the index is

For each word of the question that appears in a chunk, BM25 adds points:

- **Rare words count more** (IDF): `pagedattention` appears in few chunks → a
  match is worth a lot; `model` appears everywhere → worth almost nothing.
- **Repetition helps, with diminishing returns** (`k1`, default 1.5): 5
  occurrences score more than 1, but not 5×.
- **Long chunks are penalized** (`b`, default 0.75): a longer chunk contains
  more words "by chance", so its score is normalized by its length.

The chunk's score = the sum over the question's words.
**TF-IDF** is the older version of the same idea without the saturation (`k1`)
and the length normalization (`b`).

**`BM25()`** — creates an empty retriever (`k1=1.5, b=0.75,
method="lucene"` by default).

**`retriever.index(tokens)`** — computes the BM25 score of every word in every
chunk, once. This is the slow part (done once, by the `index` command). After
this, a search is just additions.

---

## 5. Saving

**`retriever.save(folder, corpus=[...])`** — writes the index into `folder`.

```
bm25_index/
├── data.csc.index.npy     ┐
├── indices.csc.index.npy  ├─ the score matrix (3 arrays = 1 sparse matrix)
├── indptr.csc.index.npy   ┘
├── vocab.index.json          word → row number
├── params.index.json         k1, b, method, number of chunks
├── corpus.jsonl              your chunks, one JSON object per line (only with corpus=)
└── corpus.mmindex.json       where each line of corpus.jsonl starts (for mmap)
```

- **The score matrix**: one row per word, one column per chunk, each cell =
  that word's score in that chunk. Almost all cells are 0, so only non-zero
  cells are stored ("sparse", CSC format) — that's why it's 3 arrays.
- **`nonoccurrence_array.index.npy`** appears only with the `bm25l`/`bm25+`
  methods (they give a small score even to absent words). Not with the default.

### `corpus=` — your chunks, saved next to the index

bm25s doesn't need your chunks to search; it stores them for you so that
search can turn "chunk number 57" back into a file path, offsets and text.
This replaces a separate chunks file.

- Must be a **list with one item per chunk, in the same order as the texts
  you tokenized**. Item 57 must be chunk 57.
- Each item must be **JSON-compatible**: `str`, `int`, `float`, `bool`,
  `None`, lists, dicts.
- **Trap:** pydantic objects or `Path` objects are **silently
  skipped** — `save` "succeeds" but `corpus.jsonl` is empty. Always convert
  with `chunk.model_dump()` (which you do) and store paths as strings.

**`chunk.model_dump()`** turns a `Chunk` into a dict; nested models become
nested dicts. One line of `corpus.jsonl` looks like:

```json
{"chunk": "## Dynamically serving LoRA ...", "source": {"file_path": "data/raw/vllm-0.10.1/docs/features/lora.md", "first_character_index": 4695, "last_character_index": 6098}}
```

After running `index`, check that `corpus.jsonl` has as many lines as you have
chunks.
