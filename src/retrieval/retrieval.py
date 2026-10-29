from dataclasses import dataclass, field
from pathlib import Path
from src.config import Config
from src.errors import RAGError
from src.text import expand_identifiers
from src.models import MinimalSearchResults, RagDataset, StudentSearchResults
from bm25s import BM25, tokenize
from Stemmer import Stemmer


@dataclass
class Retriever:
    config: Config
    stemmer: Stemmer
    retriever: BM25 | None = field(default=None, init=False)

    def load(self) -> BM25:
        """Load the index the first time it is needed (lazy loading)."""
        if self.retriever is None:
            path = Path(self.config.bm25_index_path)
            if not (path / "params.index.json").is_file():
                raise RAGError(
                    f"no index found in '{path}', run the index command first")
            try:
                self.retriever = BM25.load(
                    str(path), load_corpus=True, mmap=True,
                    show_progress=False)
            except (OSError, ValueError) as e:
                raise RAGError(
                    f"the index in '{path}' is unreadable ({e}), "
                    "run the index command again") from e
        return self.retriever

    def retrieve(
            self,
            querys: RagDataset,
            k: int
    ) -> StudentSearchResults:
        questions = querys.rag_questions
        bm25 = self.load()
        # bm25s raises if k is larger than the number of chunks
        top_k = min(k, bm25.scores["num_docs"])
        if top_k <= 0 or not questions:
            return StudentSearchResults(search_results=[
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=[])
                for q in questions], k=k)

        many = len(questions) > 1
        querys_tokens = tokenize(
            [expand_identifiers(q.question) for q in questions],
            stopwords="en",
            stemmer=self.stemmer,
            show_progress=many
            )

        results, scores = bm25.retrieve(
            querys_tokens, bm25.corpus, top_k, show_progress=many
            )

        search_results: list[MinimalSearchResults] = []
        for query, row, row_scores in zip(
                questions, results.tolist(), scores.tolist()):
            search_results.append(MinimalSearchResults(
                question_id=query.question_id,
                question=query.question,
                retrieved_sources=[
                    item["source"]
                    for item, score in zip(row, row_scores) if score > 0],
            ))
        return StudentSearchResults(search_results=search_results, k=k)
