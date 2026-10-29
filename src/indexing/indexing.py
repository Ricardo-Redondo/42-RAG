from pathlib import Path
from .chunking import Chunking, Chunk
from src.config import Config
from src.errors import RAGError
from dataclasses import dataclass
from bm25s import BM25, tokenize
from Stemmer import Stemmer
from tqdm import tqdm
import sys


@dataclass
class Indexer:
    config: Config
    indexer: BM25
    stemmer: Stemmer

    def index(self, strats: dict[Path, type[Chunking]]) -> int:
        """Chunk every file, build the BM25 index, save it.

        Returns the number of chunks indexed.
        """
        chunks: list[Chunk] = []
        for path, strat in tqdm(strats.items(), desc="Chunking",
                                unit="file", file=sys.stderr):
            try:
                chunks += strat.proccess(path, self.config.max_chunk_size)
            except (OSError, UnicodeDecodeError) as e:
                print(f"Warning: skipping '{path}': {e}", file=sys.stderr)
        if not chunks:
            raise RAGError(
                f"nothing to index in '{self.config.raw_path}'")

        tokens = tokenize(
            [chunk.chunk for chunk in chunks],
            stopwords="en",
            stemmer=self.stemmer
            )

        self.indexer.index(tokens)
        try:
            self.indexer.save(
                self.config.bm25_index_path,
                corpus=[chunk.model_dump() for chunk in chunks]
                )
        except OSError as e:
            raise RAGError(
                f"cannot save the index to '{self.config.bm25_index_path}': "
                f"{e.strerror}") from e
        return len(chunks)
