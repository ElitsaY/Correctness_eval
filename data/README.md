# Data

## `CAP-Correctness.csv`

QA correctness-judgment dataset used to evaluate the **CAP** (Correctness via Answer Premise/hypothesis) metric against NLI models and standard NLG metrics (BLEU, ROUGE-L, METEOR, BERTScore, COMET).

Each row is one (question, gold answer, candidate answer) triple, converted into a premise/hypothesis pair, together with a human-assigned semantic correctness label.

- **8,827 rows**
- Source QA datasets (`dataset` column): `ARC-Easy` (4,292), `openbookqa` (3,162), `mmlu` (1,373)

### Columns

| Column | Description |
|---|---|
| `question` | The original QA question. |
| `dataset` | Source QA benchmark the question was drawn from (`ARC-Easy`, `openbookqa`, `mmlu`). |
| `gold_answer` | The reference/correct answer for the question. |
| `answer` | The candidate answer being judged (may be correct, partially correct, or wrong). |
| `hypothesis` | Declarative statement generated from `question` + `gold_answer` (via [`qa_statement_generation.py`](../src/README.md), used as the NLI hypothesis / premise pair member). |
| `premise` | Declarative statement generated from `question` + `answer`, paired with `hypothesis` for NLI scoring. |
| `label_semantic` | Automatically/primarily assigned semantic correctness label (see taxonomy below). |
| `label_semantic_manual` | Manual re-annotation of `label_semantic` for a subset of rows; empty string when no manual label was assigned (7,189 of 8,827 rows). |

### Label taxonomy (`label_semantic`, `label_semantic_manual`)

Ordered from most to least correct, with the severity weight used when correlating metric scores against human judgment (see [`src/eval_comet.py`](../src/eval_comet.py) / [`src/eval_metrics.py`](../src/eval_metrics.py)):

| Label | Meaning | Severity |
|---|---|---|
| `exact` | Candidate answer matches the gold answer exactly. | 7 |
| `equivalent` | Candidate answer is a paraphrase/equivalent of the gold answer. | 6 |
| `alternative_correct` | Candidate answer is a different but also-correct answer. | 6 |
| `overinclusive_valid` | Candidate answer includes the gold answer plus extra, still-valid information. | 5 |
| `partial` | Candidate answer is partially correct. | 4 |
| `overinclusive_invalid` | Candidate answer includes the gold answer plus extra, invalid information. | 3 |
| `invalid` | Candidate answer is wrong/unsupported. | 1 |
| `contradictory` | Candidate answer contradicts the gold answer. | 0 |

Note: `cap.py` uses a slightly different severity scale (`overinclusive_invalid`: 2, `exact`: 7) — check the script being used before comparing severity-weighted results across scripts.

### Class sizes (`label_semantic`)

`overinclusive_valid` 1,182 · `alternative_correct` 1,161 · `equivalent` 1,157 · `invalid` 1,156 · `overinclusive_invalid` 1,155 · `exact` 1,045 · `contradictory` 1,031 · `partial` 940
