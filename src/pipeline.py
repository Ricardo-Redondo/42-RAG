from src.indexing import Chunking, PyChunking, MdChunking, Indexer 
from src.retrieval import Retriever
from src.config import Config
from src.models import UnansweredQuestion, RagDataset, StudentSearchResults
from pathlib import Path
from bm25s import BM25
from Stemmer import Stemmer
from pydantic import ValidationError
import sys


class Pipeline:
    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.stemmer: Stemmer = Stemmer("en")
        self.indexer: Indexer = Indexer(self.config, BM25(), self.stemmer)
        self.retriever: Retriever = Retriever(self.config, self.stemmer)

    def index(self) -> None:
        extentions = [".txt", ".py", ".md"]
        self.strats: dict[Path, Chunking] = {}

        for path in Path(self.config.raw_path).rglob("*"):
            if path.is_file() and path.suffix in extentions:
                self.strats[path] = PyChunking if path.suffix == ".py" else MdChunking
        self.indexer.index(self.strats)

    def search(self, query: UnansweredQuestion,
               k: int) -> StudentSearchResults:
        return self.retriever.retrieve(
            RagDataset(rag_questions=[query]), k)

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
        dataset = Path(dataset_path)
        try:
            querys = RagDataset.model_validate_json(
                dataset.read_text(encoding="utf-8")
                )
        except OSError as e:
            print(f"Error: cannot read dataset '{dataset_path}': "
                  f"{e.strerror}", file=sys.stderr)
            return
        except UnicodeDecodeError:
            print(f"Error: dataset '{dataset_path}' is not valid UTF-8",
                  file=sys.stderr)
            return
        except ValidationError as e:
            print(f"Error: '{dataset_path}' is not a valid dataset: "
                  f"{e.errors()[0]['msg']}", file=sys.stderr)
            return

        results = self.retriever.retrieve(querys, k)

        out_dir = Path(save_directory)
        out_file = out_dir / dataset.name
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file.write_text(results.model_dump_json(indent=2),
                                encoding="utf-8")
        except OSError as e:
            print(f"Error: cannot write '{out_file}': {e.strerror}",
                  file=sys.stderr)
            return
        print(f"Saved search results to {out_file}")

    def answer(self, query: str, k: int) -> None:
        ...