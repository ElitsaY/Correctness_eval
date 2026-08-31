# Source

Scripts used to generate the NLI premise/hypothesis pairs, compute the CAP metric, and benchmark CAP against standard NLG/NLI metrics on [`../data/CAP-Correctness.csv`](../data/README.md).

> **Note:** several scripts were extracted from notebooks and have empty/placeholder path variables (e.g. `CSV_PATH = ""`, `model_path = ""`) and use notebook-only functions like `display(...)`. Fill in the paths (typically pointing at `../data/CAP-Correctness.csv`) and, when running as a plain script rather than a notebook, replace `display(df)` calls with `print(df)` before running.

## Pipeline order

1. **`qa_statement_generation.py`** — fine-tunes the statement-generation model.
2. **`qa_inference_generation.py`** — uses that model to generate `premise`/`hypothesis` columns for new data.
3. **`cap.py`** — computes the CAP score from `premise`/`hypothesis` pairs and correlates it with human labels.
4. **`eval_metrics.py`** / **`eval_comet.py`** — benchmark CAP against BLEU/ROUGE-L/METEOR/BERTScore/COMET.

## Files

### `qa_statement_generation.py`
Fine-tunes a `google/mt5-small` sequence-to-sequence model to convert a `question` + `answer` pair into a single declarative `statement` (e.g. "Which device produces light?" + "a laser" → "A laser produces light."). Reads a training CSV with `question`, `answer`, `statement`, `language` columns, splits it into train/validation/test (stratified by `language`), tokenizes, trains with `Seq2SeqTrainer` (BLEU/ROUGE-L as validation metrics, early stopping), and evaluates on the held-out test set.

### `qa_inference_generation.py`
Loads a fine-tuned statement-generation model (the output of `qa_statement_generation.py`) and runs it over a QA CSV (`question`, `dataset`, `gold_answer`, `answer`, `label_semantic`, `label_semantic_manual`) to generate:
- `premise` — the declarative statement for `question` + `answer` (the candidate)
- `hypothesis` — the declarative statement for `question` + `gold_answer` (the reference)

Produces the enriched dataframe used as input to `cap.py` and the eval scripts — this is how [`../data/CAP-Correctness.csv`](../data/CAP-Correctness.csv) was built.

### `cap.py`
Computes the **CAP (Correctness via Answer Premise/hypothesis)** metric using an NLI model (`cross-encoder/nli-deberta-v3-large`) run over `premise`/`hypothesis` pairs. Contains two variants:
- A single-direction score: `cap_definition1 = min(1, p_entailment + 0.3 * p_neutral)` from `premise → hypothesis`.
- A bidirectional, weighted score: `cap_definition1 = 0.85 * (forward) + 0.15 * (reverse)`, combining `premise → hypothesis` and `hypothesis → premise` NLI probabilities.

Also runs the full evaluation suite against `label_semantic`: Spearman/Kendall correlation, pairwise ranking accuracy (overall and on "hard" adjacent-label pairs), pairwise AUC between classes, monotonicity-violation checks, and bootstrapped 95% confidence intervals for all of the above.

### `eval_metrics.py`
Benchmarks standard NLG metrics — **BLEU**, **ROUGE-L**, **METEOR**, **BERTScore** — against the human `label_semantic` labels, computing the same evaluation suite as `cap.py` (per-class descriptive stats, Spearman/Kendall correlation with label severity, pairwise ranking accuracy, hard-pair accuracy, monotonicity violations, bootstrapped CIs) so results are directly comparable to CAP.

### `eval_comet.py`
Same evaluation suite as `eval_metrics.py`, but for the **COMET** metric (`Unbabel/wmt22-comet-da`), scored on `question` (src) / `answer` (mt) / `gold_answer` (ref).

## Requirements

```
pandas
numpy
scipy
torch
transformers
datasets
scikit-learn
tqdm
nltk
rouge-score
bert-score
sacrebleu
unbabel-comet
```

`eval_metrics.py` also requires NLTK's `punkt` and `wordnet` corpora (`nltk.download("punkt")`, `nltk.download("wordnet")`).
