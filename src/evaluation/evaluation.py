from src.models import AnsweredQuestion, MinimalSource, RagDataset, \
    StudentSearchResults

MIN_IOU = 0.05  # overlap bar used by the moulinette


def iou(a: MinimalSource, b: MinimalSource) -> float:
    """Intersection over union of two character ranges (0 if no overlap)."""
    overlap = (min(a.last_character_index, b.last_character_index)
               - max(a.first_character_index, b.first_character_index))
    if overlap <= 0:
        return 0.0
    union = ((a.last_character_index - a.first_character_index)
             + (b.last_character_index - b.first_character_index)
             - overlap)
    return overlap / union


def is_found(truth: MinimalSource, retrieved: list[MinimalSource]) -> bool:
    """A true source is found if a result is in the same file and overlaps."""
    return any(r.file_path == truth.file_path and iou(r, truth) >= MIN_IOU
               for r in retrieved)


def recall_at_k(results: StudentSearchResults, truth: RagDataset,
                ks: list[int]) -> dict[int, float]:
    """
    Average recall@k over the questions of the ground-truth dataset.

    For one question, recall@k = share of its true sources found in the
    first k retrieved sources. Questions missing from the results count 0.
    """
    retrieved = {r.question_id: r.retrieved_sources
                 for r in results.search_results}
    questions = [q for q in truth.rag_questions
                 if isinstance(q, AnsweredQuestion) and q.sources]
    if not questions:
        return {k: 0.0 for k in ks}

    totals = {k: 0.0 for k in ks}
    for q in questions:
        found = retrieved.get(q.question_id, [])
        for k in ks:
            hits = sum(is_found(s, found[:k]) for s in q.sources)
            totals[k] += hits / len(q.sources)
    return {k: total / len(questions) for k, total in totals.items()}
