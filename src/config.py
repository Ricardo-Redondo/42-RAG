from pydantic import BaseModel, Field, ConfigDict


class Config(BaseModel):
    """
    Configuration for RAG pipeline

    model_config dictates the behavior of class:
        validate_assignment: validates values given after creation
        extra="forbid": unknown field names raise an error (typos, etc...)

    attributes:
        raw_path (str): path to the raw repo to be processed
        processed_path (str): folder where the index is saved

        max_chunk_size (int): maximun char length for a chunk

        llm (str): Hugging Face name of the model used for answers
        max_new_tokens (int): maximum length of one answer, in tokens
        max_context_sources (int): maximum number of retrieved sources
            given to the model
        max_context_tokens (int): token budget for the sources in the
            prompt (sources are cut to fit)
    """
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    raw_path: str = Field(default="data/raw/vllm-0.10.1")
    processed_path: str = Field(default="data/processed")

    max_chunk_size: int = Field(default=2000, ge=100, le=2000)

    llm: str = Field(default="Qwen/Qwen3-0.6B")
    max_new_tokens: int = Field(default=256, ge=1)
    max_context_sources: int = Field(default=5, ge=1)
    max_context_tokens: int = Field(default=1200, ge=100)

    @property
    def bm25_index_path(self) -> str:
        """Folder of the BM25 index, inside processed_path."""
        return f"{self.processed_path}/bm25_index"
