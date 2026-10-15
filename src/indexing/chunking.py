from __future__ import annotations
from abc import ABC, abstractmethod
from src.models import MinimalSource
from pathlib import Path
from pydantic import BaseModel


class Chunk(BaseModel):
    chunk: str
    source: MinimalSource


class Chunking(ABC):
    @staticmethod
    @abstractmethod
    def proccess(path: Path, chunking_size: int) -> list[Chunk]:
        ...


class PyChunking(Chunking):
    @staticmethod
    def proccess(path: Path, chunking_size: int) -> list[Chunk]:
        with open(path, newline="", encoding="utf-8") as f:
            content = f.read()
        chunks: list[Chunk] = []
        for start in range(0, len(content), chunking_size):
            end = min(start + chunking_size, len(content))
            chunks.append(Chunk(
                chunk=content[start:end],
                source=MinimalSource(
                    file_path=path.as_posix(),
                    first_character_index=start,
                    last_character_index=end
                    )
            ))
        return chunks


class MdChunking(Chunking):
    @staticmethod
    def proccess(path: Path, chunking_size: int) -> list[Chunk]:
        with open(path, newline="", encoding="utf-8") as f:
            content = f.read()
        chunks: list[Chunk] = []
        for start in range(0, len(content), chunking_size):
            end = min(start + chunking_size, len(content))
            chunks.append(Chunk(
                chunk=content[start:end],
                source=MinimalSource(
                    file_path=path.as_posix(),
                    first_character_index=start,
                    last_character_index=end
                    )
            ))
        return chunks
