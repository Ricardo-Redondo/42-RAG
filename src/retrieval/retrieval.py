from dataclasses import dataclass, field
from src.config import Config
from src.models import MinimalSearchResults, RagDataset, StudentSearchResults
from bm25s import BM25, tokenize
from Stemmer import Stemmer


@dataclass
class Retriever:
    config: Config
    stemmer: Stemmer
    retriever: BM25 | None = field(default=None, init=False)

    def retrieve(
            self,
            querys: RagDataset,
            k: int
    ) -> StudentSearchResults:
        if self.retriever is None:
            self.retriever = BM25.load(
                self.config.bm25_index_path, load_corpus=True, mmap=True
                )
        querys_tokens = tokenize(
            [q.question for q in querys.rag_questions],
            stopwords="en",
            stemmer=self.stemmer
            )

        results, _ = self.retriever.retrieve(
            querys_tokens, self.retriever.corpus, k
            )

        search_results: list[MinimalSearchResults] = []
        for query, row in zip(querys.rag_questions, results.tolist()):
            search_results.append(MinimalSearchResults(
                question_id=query.question_id,
                question=query.question,
                retrieved_sources=[item["source"] for item in row],
            ))
        return StudentSearchResults(search_results=search_results, k=k)
