# `src/` — how the program fits together

This file explains the top level of the project: how a command travels from
your terminal to the code that does the work. The details of each stage are in
`indexing/README.md` and `retrieval/README.md`.

---

## 1. The big picture

RAG = **R**etrieval-**A**ugmented **G**eneration. Four stages:

```
 data/raw/vllm-0.10.1/           (the corpus: .py and .md files)
          │
          ▼
 ┌──────────────┐   chunks    ┌──────────────┐   saved to disk
 │   CHUNKING   │ ──────────▶ │   INDEXING   │ ─────────────────▶ data/processed/bm25_index/
 └──────────────┘             └──────────────┘
                                                                        │ loaded later
 question ──────────────────────────────────────────────────────────▶   ▼
                                                              ┌──────────────┐
                                                              │  RETRIEVAL   │ ─▶ top-k sources
                                                              └──────────────┘
                                                                        │
                                                                        ▼
                                                              ┌──────────────┐
                                                              │  GENERATION  │ ─▶ answer (Qwen)
                                                              └──────────────┘
```

What is graded most: **retrieval**. For each question you return k locations
`(file_path, first_character_index, last_character_index)`. A location counts
as a hit if it is in the exact same file and overlaps the real answer's range.

---

## 2. Files in this folder

| File | Role |
|---|---|
| `__main__.py` | Entry point. Runs when you type `python -m src`. Only starts Fire. |
| `cli.py` | `Cli` class. One method per command of the subject. Thin: reads arguments, calls the pipeline, prints/saves. |
| `pipeline.py` | `Pipeline` class. The real logic: connects the indexer and the retriever. Also what the HTTP server will use. |
| `config.py` | `Config` pydantic model: all settings (paths, chunk size, model name) in one validated object. |
| `models.py` | The pydantic models from the subject (`MinimalSource`, `StudentSearchResults`, ...). |
| `indexing/` | Chunking + building/saving the BM25 index. |
| `retrieval/` | Loading the index + searching it. |
| `generation/` | (not done yet) Qwen answers. |

### Why the CLI and the pipeline are separate

The CLI deals with *the terminal* (string arguments, printing, error
messages). The pipeline deals with *the work* (returns objects, raises
exceptions). The HTTP server reuses the same pipeline with a different
"front": that is why the logic must not live in `cli.py`.

```
terminal ──▶ Cli ─┐
                  ├──▶ Pipeline ──▶ Indexer / Retriever
browser  ──▶ API ─┘
```

---

## 3. What happens when you run a command

```sh
uv run python -m src search "How to configure OpenAI server?" --k 10
```

1. **`uv run`** makes sure the `.venv` is in sync with `uv.lock`, then runs
   the rest of the line with the venv's Python.
2. **`python -m src`** runs the *package* `src` as a program. Python looks for
   `src/__main__.py` and executes it.
3. **`__main__.py`** calls `Fire(Cli)`.
4. **Fire** reads the remaining arguments:
   - creates `Cli(...)` — any `--flag` matching an `__init__` parameter goes
     there (e.g. `--raw_path`);
   - takes the first word (`search`) as the method name;
   - passes the rest to that method: `"How to configure..."` → `query`,
     `--k 10` → `k`.
5. **`Cli.__init__`** builds the `Config` and the `Pipeline`.
6. **`Cli.search`** calls the pipeline and prints the result.
7. The process **ends**. Everything in memory is gone.

### Every command is a separate process

`index` and `search` never run in the same process. That is why the index
has to be **saved to disk** by `index` and **loaded from disk** by `search`.
Nothing in memory survives between two commands.

The HTTP server is the exception: one long-running process, so it loads the
index once and keeps it for every request.

---

## 4. Functions and tools used at this level

### `fire.Fire(Cli)`

Turns a class into a command line.

- Every **public method** (no leading `_`) becomes a command.
- `__init__` parameters become flags usable with every command. They must be
  **named** parameters: `**kwargs` in `__init__` would swallow *every* flag,
  including the method's own (`--k` would never reach `search`).
- **Fire guesses types from the text**, it ignores your type hints:
  `--k 5` → `int`, `--k 5.5` → `float`, `--k abc` → `str`. That is why the
  values must be validated (pydantic) before use.
- If the method returns a value, Fire prints it. For objects (like pydantic
  models) Fire shows the object's members instead of printing it nicely, so
  print explicitly in the CLI.
- Wrong command / missing argument: Fire prints usage and exits, no traceback.


### pydantic `BaseModel` (used by `Config` and `models.py`)

- `Model(field=value, ...)` — **keyword arguments only**. Positional
  arguments raise `TypeError`.
- Values are **validated and converted** on creation: `"2000"` → `2000` for
  an `int` field; `"abc"` → `ValidationError`.
- Nested models accept **dicts**: a field typed `List[MinimalSource]` accepts a
  list of dicts with the right keys and turns each into a `MinimalSource`.
- `model_dump()` → the object as a plain dict (nested models become nested
  dicts). `model_dump_json()` → the same as a JSON string, ready to write to a
  file.
- `Model.model_validate(d)` / `Model(**d)` → build the model back from a dict.

### `ConfigDict` (in `config.py`)

`model_config = ConfigDict(...)` changes how pydantic behaves for that model:

- `validate_assignment=True` — `config.max_chunk_size = 99999` is validated
  too (by default only the constructor validates).
- `extra="forbid"` — unknown field names raise an error instead of being
  silently ignored (catches typos).

### `ValidationError`

Raised by pydantic when data is invalid. `e.errors()` is a list of dicts; each
has `loc` (which field) and `msg` (what is wrong). The edge-case script fails
on *any* traceback, so these must be caught.

### Type hints are not checked at runtime

`def f(root: Path)` does **not** convert or check anything when the program
runs: if you pass a `str`, `root` is a `str`. Only **mypy** (`make lint`)
checks hints, and only **pydantic** converts values (and only for its own
model fields).

---

## 5. Data on disk

```
data/
├── raw/vllm-0.10.1/                  input corpus (never modified)
├── datasets/public|private/
│   ├── UnansweredQuestions/*.json    questions only  → input of search_dataset
│   └── AnsweredQuestions/*.json      questions + true sources → for evaluate / moulinette
├── processed/bm25_index/             written by `index`, read by `search`
└── output/
    ├── search_results/...            written by search_dataset (StudentSearchResults)
    └── search_results_and_answer/... written by answer_dataset
```

`file_path` in every output must be exactly `data/raw/vllm-0.10.1/...` —
relative to the project root, with `/` separators.

---

## To Do

- Http Server (Bonus)
- Generation of answers to the querys using the retrieved chunks
