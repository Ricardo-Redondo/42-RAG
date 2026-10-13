from dataclasses import dataclass
from src.indexing import Chunking
from src.config import Config
from pathlib import Path


@dataclass()
class Pipeline:
    config: Config

    def __post_init__(self) -> None:
        self.strats: dict[Path, Chunking] = Chunking.select_strat(
            Path(self.config.raw_path)
            )

    def index(self):
        ...

    def search(self, query: str, k: int):
        ...

    def answer(self, query: str, k: int):
        ...