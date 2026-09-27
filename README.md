# EMNLP 2026 — How Correct Is Your Answer?

Data and code for **CAP**, an NLI-based metric for judging QA answer correctness, benchmarked against standard NLG metrics (BLEU, ROUGE-L, METEOR, BERTScore, COMET) on questions from ARC-Easy, OpenBookQA, and MMLU.

## How CAP works

A fine-tuned mT5 model turns *(question, gold answer)* into a declarative **premise** and *(question, candidate answer)* into a **hypothesis**. An NLI cross-encoder then scores both directions:

```
dir(a → b) = P(entailment | a, b) + 0.3 · P(neutral | a, b)
CAP        = 0.85 · dir(premise → hypothesis) + 0.15 · dir(hypothesis → premise)
```

The weights, models and all other settings live in [`configs/default.yaml`](configs/default.yaml).

## Repository structure

```
.
├── configs/default.yaml        # every setting for every step (seed, models + pinned revisions, weights, ...)
├── data/
│   ├── CAP-Correctness.csv     # 8,827 QA rows with correctness labels and premise/hypothesis statements
│   └── CAP-Statements.csv      # 11,000 question/answer/statement examples for the statement model
├── src/cap_eval/
│   ├── cli.py                  # `cap-eval` command line
│   ├── config.py               # typed, strictly validated config
│   ├── data.py                 # loading + validation, stable row ids
│   ├── statements.py           # statement model: training and generation
│   ├── cap.py                  # the CAP metric (NLI in both directions)
│   ├── baselines.py            # BLEU, ROUGE-L, METEOR, BERTScore, COMET
│   ├── stats.py                # correlation, pairwise ranking, AUC, bootstrap
│   ├── evaluation.py           # benchmark tables
│   └── reproducibility.py      # seeding, run manifests
├── scripts/reproduce.sh        # end-to-end reproduction
└── tests/
```

## Setup

Python ≥ 3.10.

```bash
pip install -e ".[models,comet,dev]"
```

The extras are `models` (statement model, CAP, BLEU/ROUGE-L/METEOR/BERTScore), `comet` (kept separate because it pins its own torch/lightning versions) and `dev` (pytest). The core install is enough to run `cap-eval evaluate` on existing score files.

## Reproducing the results

From the repository root:

```bash
scripts/reproduce.sh
```

This scores the released `CAP-Correctness.csv` with every metric and writes the benchmark to `outputs/evaluation/`. `scripts/reproduce.sh --full` also retrains the statement model and regenerates the premise/hypothesis statements first.

The steps can also be run individually:

```bash
cap-eval train-statements                                             # -> outputs/statement_model/final
cap-eval generate-statements --model-dir outputs/statement_model/final  # -> outputs/statements.csv
cap-eval score                                                        # -> outputs/scores/<metric>.csv
cap-eval score --metrics cap bertscore                                # just some metrics
cap-eval evaluate                                                     # -> outputs/evaluation/
```

Any config value can be overridden without editing the file, e.g. `cap-eval --set device=cuda --set cap.batch_size=64 score`. Overrides are recorded in the run manifest.

### Outputs of `evaluate`

| File | Contents |
|---|---|
| `summary.md` | Headline tables |
| `correlations.csv` | Spearman ρ, Kendall τ-b and pairwise ranking accuracy per metric, with bootstrap CIs |
| `descriptives.csv` | Per-label score distribution per metric |
| `class_pairs.csv` | For every (better, worse) label pair: accuracy, AUC, violation rate, class means |
| `hard_pairs.csv` | The subset of `class_pairs.csv` listed in `labels.hard_pairs` |
| `monotonicity.csv` | How many class pairs have their mean scores in the wrong order |
| `paired_differences.csv` | Paired-bootstrap difference between CAP and each baseline |

**Definitions.** Severity follows `labels.severity`. A pair of examples is comparable when their severities differ. *Pairwise accuracy* is the share of comparable pairs where the more-correct answer scores strictly higher (score ties count as not correct). *AUC* counts ties as half. Confidence intervals are percentile bootstrap intervals (10,000 resamples of rows). Every metric uses the same resamples, which is what makes the paired differences valid.

## Reproducibility

- **One config per run.** Every setting is in the YAML config, and unknown or missing keys are rejected.
- **Pinned models.** The Hugging Face models are pinned to commit hashes (`revision`).
- **Seeding.** Python, NumPy and PyTorch are seeded from `seed`, and so is the bootstrap. Set `statement_model.training.full_determinism: true` for deterministic CUDA kernels during training.
- **Same rows for every metric.** Rows with a missing question, answer or statement are dropped once, when the data is loaded. `row_id` is the row's position in the original CSV and is used to join score files.
- **Run manifests.** Every command writes a `*.manifest.json` next to its outputs. It records the resolved config, the command line, the git commit and whether the working tree was dirty, the SHA-256 of each input file, package versions and GPU. `evaluate` refuses to run on score files computed from a different input file.
- **Recorded splits.** The statement model's train/validation/test assignment is saved to `splits.csv`.

## Data

`CAP-Correctness.csv` has one row per (question, gold answer, candidate answer) triple. Each row has a human-assigned correctness label (`exact`, `equivalent`, `alternative_correct`, `overinclusive_valid`, `partial`, `overinclusive_invalid`, `invalid`, `contradictory`) and the generated `premise`/`hypothesis` statements. The questions come from ARC-Easy, OpenBookQA and MMLU. One row has no candidate answer and is excluded from evaluation.

`CAP-Statements.csv` has 11k examples used to train the question + answer → declarative statement model.

See [`DATA_LICENSES.md`](DATA_LICENSES.md) for licensing and attribution of the underlying QA benchmarks.

## Tests

```bash
pytest
```

The pairwise statistics are checked against brute-force implementations.

## License

Code is released under the [MIT License](LICENSE). Data licensing (including third-party attribution for ARC-Easy, OpenBookQA, and MMLU) is described in [`DATA_LICENSES.md`](DATA_LICENSES.md).
