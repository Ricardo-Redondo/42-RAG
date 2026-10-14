from __future__ import annotations
from abc import ABC, abstractmethod
from src.models import MinimalSource
from pathlib import Path
from pydantic import BaseModel


class Chunk(BaseModel):
    chunk: str
    source: MinimalSource


class Chunking(ABC):
    @abstractmethod
    def proccess(self, path: Path, chunking_size: int) -> list[Chunk]:
        ...


class PyChunking(Chunking):
    def proccess(self, path: Path, chunking_size: int) -> list[Chunk]:
        with open(path, newline="", encoding="utf-8") as f:
            content = f.read()
        chunks: list[Chunk] = []
        for i in range(chunking_size, len(content)):
            chunks.append(Chunk(
                content[i - chunking_size:i],
                MinimalSource(path.as_posix(), i - chunking_size, i)
            ))
        return chunks


class MdChunking(Chunking):
    def proccess(self, path: Path, chunking_size: int) -> list[Chunk]:
        with open(path, newline="", encoding="utf-8") as f:
            content = f.read()
        chunks: list[Chunk] = []
        for i in range(chunking_size, len(content)):
            chunks.append(Chunk(
                content[i - chunking_size:i],
                MinimalSource(path.as_posix(), i - chunking_size, i)
            ))
        return chunks
