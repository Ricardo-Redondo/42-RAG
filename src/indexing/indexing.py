from pathlib import Path
from .chunking import Chunking, Chunk
from src.config import Config
from dataclasses import dataclass
from bm25s import BM25, tokenize
from Stemmer import Stemmer


@dataclass
class Indexer:
    config: Config
    indexer: BM25
    stemmer: Stemmer

    def index(self, strats: dict[Path, Chunking]):
        chunks: list[Chunk] = []
        for path, strat in strats.items():
            chunks += strat.proccess(path, self.config.max_chunk_size)

        tokens = tokenize(
            [chunk.chunk for chunk in chunks],
            stopwords="en",
            stemmer=self.stemmer
            )

        self.indexer.index()
        self.indexer.save(
            self.config.bm25_index_path,
            corpus=[chunk.model_dump() for chunk in chunks]
            )


