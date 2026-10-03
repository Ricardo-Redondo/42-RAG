from pydantic import BaseModel, Field


class Config(BaseModel):
    """
    Configuration for RAG pipeline

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

    raw_path: str = Field(default="data/raw/vllm-0.10.1")
    datasets_path: str = Field(default="data/datasets")
    processed_path: str = Field(default="data/processed")
    chunks_output_path: str = Field(default="data/processed/chunks")
    retrieval_index_path: str = Field(default="data/processed/bm25_index")
    output_search_path: str = Field(default="data/output/search_results")
    output_answer_path: str = Field(
        default="data/output/search_results_and_answer")

    max_chunk_size: int = Field(default=2000, ge=200, le=2000)
