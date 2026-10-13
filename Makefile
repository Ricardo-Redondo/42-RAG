SGOINFRE = /sgoinfre/$(USER)
ON_42 := $(shell test -d /sgoinfre && echo yes)

ifeq ($(ON_42),yes)
export UV_PROJECT_ENVIRONMENT = $(SGOINFRE)/venvs/rag
export UV_CACHE_DIR = $(SGOINFRE)/uv-cache
export UV_LINK_MODE = copy
export HF_HOME = $(SGOINFRE)/huggingface
endif


# Commands and paths

RUN = uv run python -m src

DATASETS = data/datasets/public
SEARCH_OUT = data/output/search_results
ANSWER_OUT = data/output/search_results_and_answer
DATASET = dataset_docs_public.json
QUERY = "How to configure OpenAI server?"

MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
	--disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean destroy lint lint-strict \
	index search search_dataset answer answer_dataset evaluate

install:
ifeq ($(ON_42),yes)
	mkdir -p $(SGOINFRE)/venvs $(UV_CACHE_DIR) $(HF_HOME)
	@if [ ! -e .venv ]; then ln -s $(UV_PROJECT_ENVIRONMENT) .venv; fi
endif
	uv sync

run:
	$(RUN)

debug:
	uv run python -m pdb -m src

clean:
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	uv run flake8 .
	uv run mypy . $(MYPY_FLAGS)

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

destroy: clean
	rm -rf .venv

index:
	$(RUN) index --max_chunk_size 2000

search:
	$(RUN) search $(QUERY) --k 10

search_dataset:
	$(RUN) search_dataset \
		--dataset_path $(DATASETS)/UnansweredQuestions/$(DATASET) \
		--save_directory $(SEARCH_OUT)/UnansweredQuestions \
		--k 10

answer:
	$(RUN) answer $(QUERY) --k 10

answer_dataset:
	$(RUN) answer_dataset \
		--student_search_results_path $(SEARCH_OUT)/UnansweredQuestions/$(DATASET) \
		--save_directory $(ANSWER_OUT)/UnansweredQuestions

evaluate:
	$(RUN) evaluate \
		--student_search_results_path $(SEARCH_OUT)/UnansweredQuestions/$(DATASET) \
		--dataset_path $(DATASETS)/AnsweredQuestions/$(DATASET) \
		--k 10
