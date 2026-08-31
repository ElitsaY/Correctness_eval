# EMNLP 2026 — How Correct Is Your Answer?

Data and code for **CAP**, an NLI-based metric for judging QA answer correctness, benchmarked against standard NLG metrics (BLEU, ROUGE-L, METEOR, BERTScore, COMET) on questions from ARC-Easy, OpenBookQA, and MMLU.

## Repository structure

```
.
├── data/
│   ├── CAP-Correctness.csv   # 8,827 QA rows with correctness labels and NLI premise/hypothesis pairs
│   └── CAP-Statements.csv    # 11,000 labeled question/answer/statement examples
├── src/
│   ├── qa_statement_generation.py   # fine-tunes the question+answer -> statement generation model
│   ├── qa_inference_generation.py   # runs the model to produce premise/hypothesis pairs
│   ├── cap.py                       # computes the CAP metric and correlates it with human labels
│   ├── eval_metrics.py              # benchmarks BLEU / ROUGE-L / METEOR / BERTScore against human labels
│   └── eval_comet.py                # benchmarks COMET against human labels
├── requirements.txt
├── LICENSE            # code license (MIT)
└── DATA_LICENSES.md   # data license and source-dataset attribution
```

## Data

`CAP-Correctness.csv` — one row per (question, gold answer, candidate answer) triple, with a human-assigned semantic correctness label (`exact`, `equivalent`, `alternative_correct`, `overinclusive_valid`, `partial`, `overinclusive_invalid`, `invalid`, `contradictory`) and the generated `premise`/`hypothesis` statements used for NLI-based scoring. Drawn from ARC-Easy, OpenBookQA, and MMLU questions.

`CAP-Statements.csv` — 11k labeled examples used to train the question+answer → declarative statement generation model.

See [`DATA_LICENSES.md`](DATA_LICENSES.md) for licensing and attribution of the underlying QA benchmarks.

## Pipeline

1. `qa_statement_generation.py` — fine-tune the statement-generation model on `CAP-Statements.csv`.
2. `qa_inference_generation.py` — generate `premise`/`hypothesis` statements for new QA data.
3. `cap.py` — compute the CAP score (NLI-based) and evaluate it against human correctness labels.
4. `eval_metrics.py` / `eval_comet.py` — benchmark CAP against BLEU/ROUGE-L/METEOR/BERTScore/COMET.

## Setup

```bash
pip install -r requirements.txt
```

Several scripts were extracted from notebooks and contain placeholder path variables (e.g. `CSV_PATH = ""`) — set these to point at the files in `data/` before running.

## License

Code is released under the [MIT License](LICENSE). Data licensing (including third-party attribution for ARC-Easy, OpenBookQA, and MMLU) is described in [`DATA_LICENSES.md`](DATA_LICENSES.md).
