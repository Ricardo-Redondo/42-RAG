# `indexing/` — from files to a searchable index

This folder does two jobs, run by the `index` command:

1. **Chunking** (`chunking.py`) — cut every file into pieces ("chunks") and
   remember exactly where each piece comes from.
2. **Indexing** (`indexing.py`) — turn the chunks into a BM25 index and save it
   to disk so `search` can use it later.

```
Pipeline.index()
   │  checks raw_path exists, finds every .py/.md/.txt file,
   │  picks a chunker class for each (by extension)
   ▼
Indexer.index(strats)                    → returns the number of chunks
   │  1. chunker.proccess(path, size)  → list[Chunk]   (every file, tqdm bar)
   │  2. expand_identifiers(path + text) + tokenize    → tokens
   │  3. BM25().index(tokens)          → word statistics
   │  4. save(dir, corpus=chunks)      → files on disk
   ▼
<processed_path>/bm25_index/             (data/processed/bm25_index by default)

"Ingestion complete! 14547 chunks from 1969 files in 4.5s. Indices saved under data/processed/"
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
- **Python**: structure comes from classes and functions (`class`, `def`).

### How: LangChain's recursive splitter

Both chunkers use `RecursiveCharacterTextSplitter` from
`langchain-text-splitters`, with a different **language** — that's the one
real difference between the two strategies:

```python
splitter = RecursiveCharacterTextSplitter.from_language(
    Language.PYTHON,           # MdChunking: Language.MARKDOWN
    chunk_size=chunking_size,  # max_chunk_size (2000)
    chunk_overlap=0,           # pieces don't overlap
    add_start_index=True,      # each piece gets its position in the file
    strip_whitespace=False,    # keep every character (see below)
)
for doc in splitter.create_documents([content]):   # content = the WHOLE file
    start = doc.metadata["start_index"]             # where this chunk starts
    end = start + len(doc.page_content)             # page_content = the chunk's text
```

**"Recursive"** = it tries a list of separators **in order of preference**,
and only falls back to the next one when a piece is still longer than
`chunk_size`:

| Language | Separators, best first |
|---|---|
| `PYTHON` | `\nclass `, `\ndef `, `\n\tdef `, `\n\n`, `\n`, `" "`, `""` |
| `MARKDOWN` | headings (`\n#{1,6} `), code fences, horizontal rules, `\n\n`, `\n`, `" "`, `""` |

So a Python chunk prefers to start at a top-level `class`/`def`, a Markdown
chunk at a heading; small neighbouring pieces are merged back together up to
`chunk_size`. (`\n\tdef ` only matches tab-indented methods; vLLM indents
with 4 spaces, so inside a class the split falls back to blank lines —
which are usually between methods anyway.)

**`strip_whitespace=False` is essential.** The default (`True`) removes spaces
and newlines at the edges of every chunk: `content[start:end]` no longer
equals the chunk text and characters are lost between chunks (measured:
14,474 gaps on the corpus). With `False`, checked on all 1969 files: **0
offset errors, 0 gaps, 0 chunks over 2000**.

**The import is inside `proccess`**, not at the top of the file: it costs
~0.6 s, and only the `index` command needs it. Python imports a module once
per process, so the 1969 files share that one import. Creating a splitter is
instant (no need to cache it).

### Choosing a chunker: measured, not guessed

Recall@5 on the **public** datasets, same BM25 settings for all:

| Chunker | Plain tokens (docs / code) | With identifier splitting (docs / code) |
|---|---|---|
| fixed 2000-char windows | 85.0 / 59.6 | 84.0 / 85.9 |
| **LangChain py + md** (kept) | 82.0 / 53.5 | **83.0 / 82.8** |
| fixed windows cut at the nearest `def`/heading | 83.0 / 57.6 | 81.0 / 85.9 |
| overlapping windows (2000 every 1000) | 87.0 / 58.6 | 85.0 / 85.9 |
| Chonkie `RecursiveChunker` | 84.0 / 56.6 | — |
| Chonkie `CodeChunker` | ✗ 588 wrong offsets, chunks > 2000 | — |

- **Bigger chunks win here**: the grader only asks for an overlap of IoU ≥
  0.05, so a big chunk that *contains* the answer is enough; "smart" cuts make
  smaller chunks and lose a little.
- **Tokenization matters far more than chunking**: identifier splitting adds
  ~25 points on code, whatever the chunker (section 3).
- LangChain was kept because it gives **two real strategies** (required by the
  subject) with recall equal to the simplest option once tokenization is
  fixed.

---

## 2. Finding the files (in `pipeline.py`)

**`Path(root).rglob("*")`** — walks `root` and **all subfolders**, yielding
every entry as a `Path`. If `root` is relative (`data/raw/vllm-0.10.1`), every
result starts with that same prefix — exactly the `file_path` format the
grader wants.

**`path.is_file()`** — `True` for files, `False` for folders.

**`path.suffix`** — the extension with the dot: `".py"`, `".md"`, `""` for
`Makefile`. Case-sensitive (`.MD` ≠ `.md`).

**`sorted(...)`** — `rglob` returns files in whatever order the filesystem
gives; sorting makes the chunk order (and so the index) identical on every
machine.

**Which extensions** (`EXTENSIONS` in `pipeline.py`): the true sources of all
the AnsweredQuestions datasets point to only three kinds of files:

| Extension | Sources pointing to it |
|---|---|
| `.py` | 200 |
| `.md` | 194 |
| `.txt` | 5 |

So exactly `.py`, `.md`, `.txt` are indexed (`.py` → `PyChunking`, the others
→ `MdChunking`). Indexing files that are never answers only adds noise and
time.

---

## 3. Tokenizing

BM25 works on **words**, not text. `tokenize` turns each chunk's text into a
list of words.

### Before tokenizing: `expand_identifiers` (in `src/text.py`)

BM25 only matches **exact** words. Code glues words together:

```
question:  "What is the maximum model length setting?"
code:      max_model_len = 2048          → one word for BM25: never matches
```

`expand_identifiers` appends the pieces of every snake_case / CamelCase name,
**keeping the original** (a question quoting the exact name still matches):

```python
expand_identifiers("TritonAttentionMetadataBuilder uses max_model_len")
# → "TritonAttentionMetadataBuilder uses max_model_len Triton Attention Metadata Builder max model len"
```

```python
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")       # anything that looks like a name
WORD_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

for name in IDENTIFIER.findall(text):
    if "_" in name or any(c.isupper() for c in name[1:]):   # only compound names
        for piece in name.split("_"):                        # snake_case
            parts += WORD_PART.findall(piece)                # CamelCase
return text + " " + " ".join(parts)
```

`WORD_PART` alternatives, tried in order: a run of capitals followed by a new
word (`HTTPServer` → `HTTP`), a word (`Triton`, `model`), an all-caps word at
the end, digits. Known quirk: `LoRARequest` → `Lo RA Request` (the original
`LoRARequest` is still there).

**The file path is added too**: each chunk is tokenized as
`expand_identifiers(file_path + " " + chunk_text)`. A chunk of
`vllm/lora/worker_manager.py` gains `vllm lora worker manager` — useful when
the chunk text itself never says what it's about.

**Only the BM25 tokens change**: the stored chunk text and its offsets are
untouched, so `content[start:end] == chunk` still holds and Qwen still reads
the real text.

Effect (recall@5, LangChain chunks):

| | Without | With |
|---|---|---|
| docs public / private | 82.0 / 81.0 | 83.0 / 82.0 |
| code public / private | 53.5 / 56.0 | **82.8 / 75.0** |

The retriever calls the **same** function on the questions — see the rule
below.

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

---

## 6. Errors and progress

| Situation | What happens |
|---|---|
| `raw_path` is not a folder | `RAGError("corpus folder '...' not found")` → exit 1 |
| one file can't be read (permissions, not UTF-8) | `Warning: skipping '...'` on stderr, indexing continues |
| no chunk at all (empty folder) | `RAGError("nothing to index in '...'")` |
| the index can't be saved (disk full, read-only) | `RAGError("cannot save the index to '...'")` |

Progress bars (the subject asks for tqdm on long operations): `Chunking`
(ours, one step per file) then bm25s's own bars (`Split strings`,
`Stem Tokens`, `BM25S Count Tokens`, `BM25S Compute Scores`). They all go to
stderr.
