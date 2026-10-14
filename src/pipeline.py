from src.indexing import Chunking, PyChunking, MdChunking, Indexer 
from src.config import Config
from pathlib import Path
from bm25s import BM25
from Stemmer import Stemmer


class Pipeline:
    def __init__(self, config: Config):
        self.config: Config = config
        self.indexer: Indexer = Indexer(config, BM25(), Stemmer("en"))

    def index(self):
        extentions = [".txt", ".py", ".md"]
        self.strats: dict[Path, Chunking] = {}

        for path in Path(self.config.raw_path).rglob("*"):
            if path.is_file() and path.suffix in extentions:
                self.strats[path] = PyChunking if path.suffix == ".py" else MdChunking
        self.indexer.index(self.strats)

    def search(self, query: str, k: int):
        ...

    def answer(self, query: str, k: int):
        ...