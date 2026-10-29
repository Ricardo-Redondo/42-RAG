# `src/` — how the program fits together

This file explains the top level of the project: how a command travels from
your terminal to the code that does the work, and how errors are handled.
The details of each stage are in `indexing/README.md`, `retrieval/README.md`,
`generation/README.md` and `evaluation/README.md`.

---

## 1. The big picture

RAG = **R**etrieval-**A**ugmented **G**eneration. Four stages:

```
 data/raw/vllm-0.10.1/           (the corpus: .py, .md and .txt files)
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
| `__main__.py` | Entry point. Runs when you type `python -m src`. Starts Fire and turns **every** error into a clean message + exit code. |
| `cli.py` | `Cli` class. One method per command of the subject. Thin: checks the arguments, calls the pipeline, prints. |
| `pipeline.py` | `Pipeline` class. The real logic: connects indexer, retriever, generator and evaluation. Reads/writes the JSON files. Also what the HTTP server will use. |
| `config.py` | `Config` pydantic model: all settings (paths, chunk size, model, prompt budget) in one validated object. |
| `models.py` | The pydantic models from the subject (`MinimalSource`, `StudentSearchResults`, ...). |
| `errors.py` | `RAGError`: an *expected* error (bad input, missing index...) whose message is meant for the user. |
| `text.py` | `expand_identifiers`: splits `max_model_len` / `CamelCase` names for BM25. Used by **both** indexing and retrieval. |
| `indexing/` | Chunking + building/saving the BM25 index. |
| `retrieval/` | Loading the index + searching it. |
| `generation/` | Turning a question + its sources into an answer with Qwen. |
| `evaluation/` | recall@k, for the `evaluate` command. |

### Why the CLI and the pipeline are separate

The CLI deals with *the terminal* (string arguments, printing). The pipeline
deals with *the work* (returns objects, **raises** exceptions, never prints
errors itself). The HTTP server reuses the same pipeline with a different
"front": that is why the logic must not live in `cli.py`.

```
terminal ──▶ Cli ─┐
                  ├──▶ Pipeline ──▶ Indexer / Retriever / Generator / recall_at_k
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
3. **`__main__.py`** calls `main()`, which calls `Fire(Cli)` inside a `try`.
4. **Fire** reads the remaining arguments:
   - creates `Cli(...)` — any `--flag` matching an `__init__` parameter goes
     there (`--raw_path`, `--processed_path`, `--llm`);
   - takes the first word (`search`) as the method name;
   - passes the rest to that method: `"How to configure..."` → `query`,
     `--k 10` → `k`.
5. **`Cli.__init__`** builds the `Config` and the `Pipeline` (which creates
   an *empty* indexer, retriever and generator: nothing is loaded yet).
6. **`Cli.search`** checks `k`, wraps the query in an `UnansweredQuestion`,
   calls the pipeline and prints the sources.
7. The process **ends**. Everything in memory is gone.

### Every command is a separate process

`index` and `search` never run in the same process. That is why the index
has to be **saved to disk** by `index` and **loaded from disk** by `search`.
Nothing in memory survives between two commands.

The HTTP server is the exception: one long-running process, so it loads the
index and the model once and keeps them for every request.

### Lazy loading and lazy importing

Each command only pays for what it uses:

| Command | Loads the index? | Imports transformers/torch + loads Qwen? |
|---|---|---|
| `index` | no (builds it) | no |
| `search`, `search_dataset` | yes, once | **no** |
| `answer`, `answer_dataset` | yes (answer only) | yes, once — skipped if no sources |
| `evaluate` | no | no |

`transformers` and `torch` are imported *inside* `Generator.load()`, not at
the top of a file: importing them alone takes 3–7 s, which `search` would
otherwise pay too (and the throughput test runs `search_dataset` twice).

---

## 4. Errors: why no command ever shows a traceback

The edge-case test fails on **any** traceback. All errors flow up to one place:

```
Retriever / Generator / Pipeline ──raise RAGError("no index found ...")──┐
CLI (check_k, pydantic models)   ──raise RAGError / ValidationError──────┤
                                                                         ▼
                              __main__.main():  except RAGError       → "Error: ..."            exit 1
                                                except ValidationError → "Invalid argument ..." exit 1
                                                except KeyboardInterrupt → message              exit 130
                                                except Exception       → "Unexpected error ..." exit 1
```

- **`RAGError`** (`errors.py`) = a problem we expected, with a message written
  for the user. Low-level errors are translated into it where they happen:
  `OSError` reading a file → `RAGError("cannot read 'x': No such file...")`.
- The last `except Exception` is a safety net: an unexpected bug still prints
  one line instead of a traceback.
- Fire's own errors (unknown command, missing argument) print the usage and
  exit by themselves — they are `SystemExit`, not caught here.

### Argument checks in `cli.py`

**Fire guesses types from the text** and ignores type hints:

| You type | Fire passes | Handled by |
|---|---|---|
| `--k 10` | `10` (int) | fine |
| `--k abc` | `"abc"` (str) | `check_k` → "--k must be an integer >= 0" |
| `--k -1` | `-1` | `check_k` → same error |
| `search 123` | `123` (int!) | `make_question` → `str(query)` |

### Edge cases (the exam's 4)

| Input | Result |
|---|---|
| `search "" --k 10` | "No relevant sources found." (score-0 results are dropped) |
| `search "asdfghjkl ..." --k 10` | "No relevant sources found." |
| `answer "What is vLLM?" --k 0` | "No relevant sources were found..." (Qwen never loaded) |
| `search_dataset --dataset_path /nonexistent.json` | "Error: cannot read ...", exit 1 |

`make edge_cases` runs the four.

---

## 5. Functions and tools used at this level

### `fire.Fire(Cli)`

Turns a class into a command line.

- Every **public method** (no leading `_`) becomes a command.
- `__init__` parameters become flags usable with every command. They must be
  **named** parameters: `**kwargs` in `__init__` would swallow *every* flag,
  including the method's own (`--k` would never reach `search`).
- **Fire guesses types from the text** (see the table above).
- If the method returns a value, Fire prints it. For objects (like pydantic
  models) Fire shows the object's members instead of printing it nicely, so
  the CLI prints explicitly and returns `None`.

### pydantic `BaseModel` (used by `Config` and `models.py`)

- `Model(field=value, ...)` — **keyword arguments only**. Positional
  arguments raise `TypeError`.
- Values are **validated and converted** on creation: `"2000"` → `2000` for
  an `int` field; `"abc"` → `ValidationError`. But an `int` is **not**
  converted to a `str` field (hence `make_question`).
- Nested models accept **dicts**: a field typed `List[MinimalSource]` accepts a
  list of dicts with the right keys and turns each into a `MinimalSource`.
- `model_dump()` → the object as a plain dict. `model_dump_json(indent=2)` →
  the same as a JSON string, ready to write to a file.
- `Model.model_validate_json(text)` → parse + validate a JSON string
  (used by `read_json` in `pipeline.py`).

### `ConfigDict` (in `config.py`)

`model_config = ConfigDict(...)` changes how pydantic behaves for that model:

- `validate_assignment=True` — `config.max_chunk_size = 99999` is validated
  too (by default only the constructor validates).
- `extra="forbid"` — unknown field names raise an error instead of being
  silently ignored (catches typos).

### The `Config` fields

| Field | Default | Used by |
|---|---|---|
| `raw_path` | `data/raw/vllm-0.10.1` | indexing (`--raw_path`) |
| `processed_path` | `data/processed` | where the index lives (`--processed_path`) |
| `bm25_index_path` | *property*: `<processed_path>/bm25_index` | indexing + retrieval |
| `max_chunk_size` | 2000 (100–2000) | chunking (`index --max_chunk_size`) |
| `llm` | `Qwen/Qwen3-0.6B` | generation (`--llm`) |
| `max_new_tokens` | 256 | max length of one answer |
| `max_context_sources` | 5 | max sources put in the prompt |
| `max_context_tokens` | 1200 | token budget of the sources in the prompt |

`bm25_index_path` is a `@property`: computed from `processed_path` every time
it's read, so `--processed_path` moves the index too.

### `read_json` / `write_json` (in `pipeline.py`)

The two helpers every dataset command uses:

- `read_json(path, Model)` → reads the file and validates it into `Model`.
  Translates `OSError`, `UnicodeDecodeError` and `ValidationError` into
  `RAGError`. It's **generic**: `TypeVar("Model", bound=BaseModel)` tells mypy
  "you get back an instance of the class you passed in".
- `write_json(save_directory, name, data)` → creates the folder if needed and
  writes `save_directory/name` (`Path / str` joins paths). The file name is the
  dataset's name (`dataset_docs_private.json`): the exam script looks for
  `*docs*` / `*code*` in the save directory.

### Type hints are not checked at runtime

`def f(root: Path)` does **not** convert or check anything when the program
runs: if you pass a `str`, `root` is a `str`. Only **mypy** (`make lint`)
checks hints, and only **pydantic** converts values (and only for its own
model fields).

---

## 6. Data on disk

```
data/
├── raw/vllm-0.10.1/                  input corpus (never modified)
├── datasets/public|private/
│   ├── UnansweredQuestions/*.json    questions only  → input of search_dataset
│   └── AnsweredQuestions/*.json      questions + true sources → for evaluate / moulinette
├── processed/bm25_index/             written by `index`, read by `search`/`answer`
└── output/
    ├── search_results/UnansweredQuestions/...            search_dataset (StudentSearchResults)
    └── search_results_and_answer/UnansweredQuestions/... answer_dataset (StudentSearchResultsAndAnswer)
```

`file_path` in every output must be exactly `data/raw/vllm-0.10.1/...` —
relative to the project root, with `/` separators.

---

## 7. Make shortcuts

| Rule | Runs |
|---|---|
| `make index` | `index --max_chunk_size $(CHUNK)` |
| `make search` / `make answer` | one question (`QUERY=...`, `K=...`) |
| `make search_dataset` / `make answer_dataset` | one dataset (`TYPE=docs\|code`, `SCOPE=public\|private`) |
| `make evaluate` (`make eval`) | our recall@k |
| `make moulinette` | the official grader on the same files |
| `make search_all` / `eval_all` / `moulinette_all` | docs + code |
| `make check` | index → search_all → eval_all (the exam's retrieval run) |
| `make edge_cases` | the exam's 4 edge cases |
| `make clean` / `fclean` / `vclean` | caches / + index and outputs / + the venv |

Example: `make check SCOPE=private K=10`. The exam always uses **k = 10**
and grades **recall@5**.

---

## 8. Current results

LangChain chunks (Python / Markdown splitters, 2000 chars) + identifier
splitting + file path words, BM25 (k1=1.5, b=0.75) with English stopwords and
stemming:

| Exam check | Result | Limit |
|---|---|---|
| indexing | ~4.5 s | 300 s |
| 200 searches | ~0.7 s | 90 s |
| docs recall@5 (public / private) | 83.0% / 82.0% | 80% |
| code recall@5 (public / private) | 82.8% / 75.0% | 50% |
| edge cases | 4 / 4, no traceback | all |
| one answer (CPU) | ~13 s + ~6 s model loading | — |

The private scores are confirmed by the moulinette. How we got there (each
change measured): `indexing/README.md` section "Choosing a chunker".

## To Do

- Docs is the closest to its line (82% vs 80%): overlapping windows were the
  only thing that raised docs (87% public) — worth re-testing with the new
  tokenization
- HTTP server (bonus)
- Fill in the login on the first line of the root `README.md`
