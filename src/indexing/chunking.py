from __future__ import annotations
from abc import ABC, abstractmethod
from src.models import MinimalSource
from pathlib import Path


class Chunking(ABC):
    @abstractmethod
    def proccess(self, path: Path) -> list[MinimalSource]:
        ...

    @staticmethod
    def select_strat(root: Path) -> dict[Path, Chunking]:
        extentions = [".txt", ".py", ".md"]
        res = {}

        for path in root.rglob("*"):
            if path.is_file() and path.suffix in extentions:
                res[path] = PyChunking if path.suffix == ".py" else MdChunking
        return res


class PyChunking(Chunking):
    def proccess(self, path: Path) -> list[MinimalSource]:
        ...


class MdChunking(Chunking):
    def proccess(self, path: Path) -> list[MinimalSource]:
        ...
