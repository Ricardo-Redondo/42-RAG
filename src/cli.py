from pathlib import Path
from src.pipeline import Pipeline
from src.config import Config


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

    def index(self, max_chunk_size: int = 2000):
        """Ingest data/raw/ and build the index under data/processed/."""
        self.config.max_chunk_size = max_chunk_size
        print("index")
        ...

    def search(self, query: str, k: int):
        """Return the top-k sources for a single query."""
        print("search")
        ...

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str):
        """
        Run search over a whole dataset
        and write a StudentSearchResults JSON file.
        """
        print("search_dataset")
        ...

    def answer(self, query: str, k: int):
        """Answer a single query using the retrieved context."""
        print("answer")
        ...

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str):
        """
        Generate answers for a dataset,
        producing a StudentSearchResultsAndAnswer
        JSON file.
        """
        print("answer_dataset")
        ...

    def evaluate(student_search_results_path: str, dataset_path: str):
        """
        Report your own recall@k against a ground-truth dataset,
        for your own testing.
        """
        print("evaluate")
        ...