# `evaluation/` — recall@k, for the `evaluate` command

The subject asks for an `evaluate` command "for your own iteration": it reports
your recall@k so you can measure every change (chunk size, stemmer, ...).
The official score at the defense comes from the **moulinette**, and the
project must never import or call it — so this is our own implementation of
the same metric.

```sh
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/public/AnsweredQuestions/dataset_docs_public.json
# Recall@1: 0.560  Recall@3: 0.770  Recall@5: 0.850  Recall@10: 0.870
```

(`make eval`, `make eval_all`.)

**It gives exactly the moulinette's numbers** — checked on both public
datasets (docs 0.560 / 0.770 / 0.850 / 0.870, code 0.354 / 0.535 / 0.596 /
0.667).

---

## 1. The metric, from the subject

> For each question, recall@k is the share of its correct sources that you
> retrieve in your top-k results. A correct source counts as found when one of
> your results is in the same file and overlaps its character range. The
> overlap bar is low (an IoU of 0.05).

### IoU (intersection over union) of two character ranges

```python
def iou(a: MinimalSource, b: MinimalSource) -> float:
    overlap = (min(a.last_character_index, b.last_character_index)
               - max(a.first_character_index, b.first_character_index))
    if overlap <= 0:
        return 0.0                         # the ranges don't touch
    union = (a_length + b_length - overlap)
    return overlap / union
```

```
true source:        [-------- 4695 .. 6098 --------]
our chunk:   [------------ 4000 .. 6000 ------------]
overlap = 6000 - 4695 = 1305      union = 2000 + 1403 - 1305 = 2098
IoU = 1305 / 2098 = 0.62  ≥ 0.05 → found
```

Because the bar is so low (0.05), a 2000-character chunk only needs to cover
a small part of the true span. A result in another file never counts — that's
why `file_path` must be exact.

### Found or not

```python
def is_found(truth, retrieved) -> bool:
    return any(r.file_path == truth.file_path and iou(r, truth) >= MIN_IOU
               for r in retrieved)
```

`any(...)` stops at the first result that matches.

### Averaging

```python
retrieved = {r.question_id: r.retrieved_sources for r in results.search_results}
for q in questions:                                   # every answered question of the truth file
    found = retrieved.get(q.question_id, [])          # missing from our results → counts 0
    for k in ks:
        hits = sum(is_found(s, found[:k]) for s in q.sources)   # True counts as 1
        totals[k] += hits / len(q.sources)            # this question's recall@k
return {k: total / len(questions) for k, total in totals.items()}
```

- Questions are matched by **`question_id`**, not by position — a results file
  in a different order still scores correctly.
- `found[:k]` — only the first k results count for recall@k. Results are
  ranked best first, so recall@1 ≤ recall@3 ≤ recall@5 ≤ recall@10.
- `sum(True, False, True)` = 2: booleans count as 1 and 0.
- A dict lookup (`retrieved.get`) instead of searching the list every time:
  one step per question instead of 100.

---

## 2. Which k are reported

`Pipeline.evaluate` reports recall@1, @3, @5 and @10 — but only the ones
**≤ the k the search was run with** (`results.k`). A search done with `K=3`
can't say anything about recall@5: there are only 3 results per question.
The exam always uses **k = 10** and grades **recall@5**.

---

## 3. Thresholds (eval sheet)

| Dataset | recall@5 to pass | stars |
|---|---|---|
| docs | ≥ 80% | +1 star per +5% (85, 90, 95) |
| code | ≥ 50% | +1 star per +5% (55, 60, 65) |
