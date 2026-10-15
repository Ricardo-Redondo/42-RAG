from pathlib import Path
from src.models import UnansweredQuestion
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

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
        self.config.max_chunk_size = max_chunk_size
        self.pipeline.index()

    def search(self, query: str, k: int) -> None:
        """Return the top-k sources for a single query."""
        result = self.pipeline.search(UnansweredQuestion(question=query), k)
        for source in result.search_results[0].retrieved_sources:
            print(f"{source.file_path} [{source.first_character_index}/"
                  f"{source.last_character_index}]")

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
        """
        Run search over a whole dataset
        and write a StudentSearchResults JSON file.
        """
        self.pipeline.search_dataset(dataset_path, k, save_directory)

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