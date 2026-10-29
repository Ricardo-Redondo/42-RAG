SGOINFRE = /sgoinfre/$(USER)
ON_42 := $(shell test -d /sgoinfre && echo yes)

ifeq ($(ON_42),yes)
export UV_PROJECT_ENVIRONMENT = $(SGOINFRE)/venvs/rag
export UV_CACHE_DIR = $(SGOINFRE)/uv-cache
export UV_LINK_MODE = copy
export HF_HOME = $(SGOINFRE)/huggingface
endif


RUN = uv run python -m src

SCOPE ?= public
TYPE ?= docs
K ?= 10
CHUNK ?= 2000
QUERY ?= How to configure OpenAI server?
MOULINETTE ?= moulinette/moulinette-ubuntu

DATASET = dataset_$(TYPE)_$(SCOPE).json
UNANSWERED = data/datasets/$(SCOPE)/UnansweredQuestions/$(DATASET)
ANSWERED = data/datasets/$(SCOPE)/AnsweredQuestions/$(DATASET)
SEARCH_OUT = data/output/search_results/UnansweredQuestions
ANSWER_OUT = data/output/search_results_and_answer/UnansweredQuestions
RESULTS = $(SEARCH_OUT)/$(DATASET)

MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
	--disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean destroy lint lint-strict \
	index search search_dataset answer answer_dataset evaluate eval \
	moulinette search_all eval_all moulinette_all check edge_cases

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
	$(RUN) index --max_chunk_size $(CHUNK)

search:
	$(RUN) search "$(QUERY)" --k $(K)

search_dataset:
	$(RUN) search_dataset --dataset_path $(UNANSWERED) \
		--k $(K) --save_directory $(SEARCH_OUT)

answer:
	$(RUN) answer "$(QUERY)" --k $(K)

answer_dataset:
	$(RUN) answer_dataset --student_search_results_path $(RESULTS) \
		--save_directory $(ANSWER_OUT)

evaluate:
	$(RUN) evaluate --student_search_results_path $(RESULTS) \
		--dataset_path $(ANSWERED)

eval: evaluate

moulinette:
	$(MOULINETTE) evaluate_student_search_results $(RESULTS) $(ANSWERED) \
		--k $(K) --max_context_length 2000


search_all:
	$(MAKE) --no-print-directory search_dataset TYPE=docs
	$(MAKE) --no-print-directory search_dataset TYPE=code

eval_all:
	$(MAKE) --no-print-directory evaluate TYPE=docs
	$(MAKE) --no-print-directory evaluate TYPE=code

moulinette_all:
	$(MAKE) --no-print-directory moulinette TYPE=docs
	$(MAKE) --no-print-directory moulinette TYPE=code

check: index search_all eval_all

edge_cases:
	-$(RUN) search "" --k 10
	-$(RUN) search "asdfghjkl zxcvbnm qwertyuiop" --k 10
	-$(RUN) answer "What is vLLM?" --k 0
	-$(RUN) search_dataset --dataset_path /nonexistent/dataset.json --k 10
