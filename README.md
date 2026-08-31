# CAP-Correctness

Data and code for **CAP**, an NLI-based metric for judging QA answer correctness, benchmarked against standard NLG metrics (BLEU, ROUGE-L, METEOR, BERTScore, COMET) on questions from ARC-Easy, OpenBookQA, and MMLU.

## Structure

- [`data/`](data/README.md) — the labeled QA correctness dataset (`CAP-Correctness.csv`) and a description of its columns and label taxonomy.
- [`src/`](src/README.md) — the pipeline scripts: statement generation, CAP scoring, and evaluation against BLEU/ROUGE-L/METEOR/BERTScore/COMET.

See each folder's README for details.
