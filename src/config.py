from pydantic import BaseModel, Field, ConfigDict


class Config(BaseModel):
    """
    Configuration for RAG pipeline

    model_config dictates the behavior of class:
        validate_assignment: validates values given after creation
        extra="fordid": validates typing (typos, etc...)

    attributes:
        raw_path (str): path to the raw repo to be processed
        datasets_path (str): path to the test datasets
        processed_path (str): path to the processed data
        chunks_output_path (str): path to the processed chunks
        retrieval_index_path (str): path to the bm25 index
        output_search_path (str): path to the outputed search results
        output_answer_path (str): path to the outputed generated answers

        max_chunk_size (int): maximun char length for a chunk
    """
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    raw_path: str = Field(default="data/raw/vllm-0.10.1")
    datasets_path: str = Field(default="data/datasets")
    processed_path: str = Field(default="data/processed")
    chunks_output_path: str = Field(default="data/processed/chunks")
    retrieval_index_path: str = Field(default="data/processed/bm25_index")
    output_search_path: str = Field(default="data/output/search_results")
    output_answer_path: str = Field(
        default="data/output/search_results_and_answer")

    max_chunk_size: int = Field(default=2000, ge=100, le=2000)
    llm: str = Field(default="Qwen/Qwen3-0.6B")
