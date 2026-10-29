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
        from langchain_text_splitters import Language, \
            RecursiveCharacterTextSplitter

        with open(path, newline="", encoding="utf-8") as f:
            content = f.read()
        splitter = RecursiveCharacterTextSplitter.from_language(
            Language.PYTHON,
            chunk_size=chunking_size,
            chunk_overlap=0,
            add_start_index=True,
            strip_whitespace=False
        )

        chunks: list[Chunk] = []
        for doc in splitter.create_documents([content]):
            start_index = doc.metadata["start_index"]
            last_index = start_index + len(doc.page_content)
            chunks.append(
                Chunk(
                    chunk=doc.page_content,
                    source=MinimalSource(
                        file_path=path.as_posix(),
                        first_character_index=start_index,
                        last_character_index=last_index
                    )
                )
            )
        return chunks


class MdChunking(Chunking):
    @staticmethod
    def proccess(path: Path, chunking_size: int) -> list[Chunk]:
        from langchain_text_splitters import Language, \
            RecursiveCharacterTextSplitter

        with open(path, newline="", encoding="utf-8") as f:
            content = f.read()

        splitter = RecursiveCharacterTextSplitter.from_language(
                    Language.MARKDOWN,
                    chunk_size=chunking_size,
                    chunk_overlap=0,
                    add_start_index=True,
                    strip_whitespace=False
                )

        chunks: list[Chunk] = []
        for doc in splitter.create_documents([content]):
            start_index = doc.metadata["start_index"]
            last_index = start_index + len(doc.page_content)
            chunks.append(
                Chunk(
                    chunk=doc.page_content,
                    source=MinimalSource(
                        file_path=path.as_posix(),
                        first_character_index=start_index,
                        last_character_index=last_index
                    )
                )
            )
        return chunks
