from src.indexing import Chunking, PyChunking, MdChunking, Indexer
from src.retrieval import Retriever
from src.generation import Generator
from src.evaluation import recall_at_k
from src.config import Config
from src.errors import RAGError
from src.models import UnansweredQuestion, RagDataset, \
    StudentSearchResults, StudentSearchResultsAndAnswer
from pathlib import Path
from bm25s import BM25
from Stemmer import Stemmer
from pydantic import BaseModel, ValidationError
from typing import TypeVar
import time

Model = TypeVar("Model", bound=BaseModel)

EXTENSIONS = [".txt", ".py", ".md"]


def read_json(path: str, model: type[Model]) -> Model:
    """Read a JSON file and validate it into a pydantic model."""
    try:
        return model.model_validate_json(
            Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise RAGError(f"cannot read '{path}': {e.strerror}") from e
    except UnicodeDecodeError as e:
        raise RAGError(f"'{path}' is not valid UTF-8") from e
    except ValidationError as e:
        raise RAGError(f"'{path}' is not a valid {model.__name__} file: "
                       f"{e.errors()[0]['msg']}") from e


def write_json(save_directory: str, name: str, data: BaseModel) -> Path:
    """Write a pydantic model as JSON to save_directory/name."""
    out_dir = Path(save_directory)
    out_file = out_dir / name
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file.write_text(data.model_dump_json(indent=2), encoding="utf-8")
    except OSError as e:
        raise RAGError(f"cannot write '{out_file}': {e.strerror}") from e
    return out_file


class Pipeline:
    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.stemmer: Stemmer = Stemmer("en")
        self.indexer: Indexer = Indexer(self.config, BM25(), self.stemmer)
        self.retriever: Retriever = Retriever(self.config, self.stemmer)
        self.generator: Generator = Generator(self.config)

    def index(self) -> None:
        root = Path(self.config.raw_path)
        if not root.is_dir():
            raise RAGError(f"corpus folder '{root}' not found")
        start = time.perf_counter()
        self.strats: dict[Path, type[Chunking]] = {}

        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in EXTENSIONS:
                self.strats[path] = (
                    PyChunking if path.suffix == ".py" else MdChunking)
        n_chunks = self.indexer.index(self.strats)
        elapsed = time.perf_counter() - start
        print(f"Ingestion complete! {n_chunks} chunks from "
              f"{len(self.strats)} files in {elapsed:.1f}s. "
              f"Indices saved under {self.config.processed_path}/")

    def search(self, query: UnansweredQuestion,
               k: int) -> StudentSearchResults:
        return self.retriever.retrieve(
            RagDataset(rag_questions=[query]), k)

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
        querys = read_json(dataset_path, RagDataset)
        results = self.retriever.retrieve(querys, k)
        out_file = write_json(
            save_directory, Path(dataset_path).name, results)
        print(f"Saved student_search_results to {out_file}")

    def answer(self, query: UnansweredQuestion,
               k: int) -> StudentSearchResultsAndAnswer:
        results = self.retriever.retrieve(
            RagDataset(rag_questions=[query]), k)
        return self.generator.answer(results)

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        search_results = read_json(
            student_search_results_path, StudentSearchResults)
        try:
            answers = self.generator.answer(search_results)
        except OSError as e:
            raise RAGError(f"cannot generate answers: {e}") from e
        out_file = write_json(
            save_directory, Path(student_search_results_path).name, answers)
        print(f"Saved student_search_results_and_answer to {out_file}")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> dict[int, float]:
        results = read_json(student_search_results_path, StudentSearchResults)
        truth = read_json(dataset_path, RagDataset)
        ks = [k for k in (1, 3, 5, 10) if k <= results.k] or [results.k]
        return recall_at_k(results, truth, ks)
