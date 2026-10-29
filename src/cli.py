from typing import Any
from src.errors import RAGError
from src.models import UnansweredQuestion
from src.pipeline import Pipeline
from src.config import Config


def check_k(k: Any) -> int:
    """Fire guesses types from the text: make sure k is an int >= 0."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise RAGError(f"--k must be an integer >= 0, got '{k}'")
    return k


def make_question(query: Any) -> UnansweredQuestion:
    """Fire turns `search 123` into an int: always use the text."""
    return UnansweredQuestion(question=str(query))


class Cli:
    def __init__(self, raw_path: str = "data/raw/vllm-0.10.1",
                 processed_path: str = "data/processed",
                 llm: str = "Qwen/Qwen3-0.6B") -> None:
        self.config = Config(
            raw_path=raw_path,
            processed_path=processed_path,
            llm=llm
            )
        self.pipeline = Pipeline(config=self.config)

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
        self.config.max_chunk_size = max_chunk_size
        self.pipeline.index()

    def search(self, query: str, k: int = 10) -> None:
        """Return the top-k sources for a single query."""
        result = self.pipeline.search(make_question(query), check_k(k))
        sources = result.search_results[0].retrieved_sources
        if not sources:
            print("No relevant sources found.")
        for source in sources:
            print(f"{source.file_path} [{source.first_character_index}/"
                  f"{source.last_character_index}]")

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/search_results"
                       ) -> None:
        """
        Run search over a whole dataset
        and write a StudentSearchResults JSON file.
        """
        self.pipeline.search_dataset(dataset_path, check_k(k),
                                     save_directory)

    def answer(self, query: str, k: int) -> None:
        """Answer a single query using the retrieved context."""
        print("answer")
        ...

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """
        Generate answers for a dataset,
        producing a StudentSearchResultsAndAnswer
        JSON file.
        """
        print("answer_dataset")
        ...

    def evaluate(student_search_results_path: str,
                 dataset_path: str) -> None:
        """
        Report your own recall@k against a ground-truth dataset,
        for your own testing.
        """
        print("evaluate")
        ...