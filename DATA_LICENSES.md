# Data Licenses

The contents of `data/` combine original annotations with questions and answers drawn from third-party QA benchmarks. Different license terms apply to each part.

## Original annotations — CC BY 4.0

The following are original contributions of this project and are released under **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**:

- `label_semantic`, `label_semantic_manual` (correctness labels) in `CAP-Correctness.csv`
- `premise`, `hypothesis` (generated declarative statements) in `CAP-Correctness.csv`
- `statement`, `label_question_type` in `CAP-Statements.csv`

If you reuse these annotations, please provide attribution to this repository.

## Source QA data — third-party licenses

The `question`, `answer`/`gold_answer`, `subject`, and `dataset` fields originate from the following benchmarks and remain subject to their original licenses:

| Source dataset | Rows drawn from it | License | Source |
|---|---|---|---|
| [ARC-Easy](https://allenai.org/data/arc) (AI2 Reasoning Challenge) | `CAP-Correctness.csv` | CC BY-SA 4.0 | Allen Institute for AI |
| [OpenBookQA](https://allenai.org/data/open-book-qa) | `CAP-Correctness.csv` | Apache License 2.0 | Allen Institute for AI |
| [MMLU](https://github.com/hendrycks/test) (Measuring Massive Multitask Language Understanding) | `CAP-Correctness.csv` | MIT License | Hendrycks et al. |

`CAP-Statements.csv` contains independently authored question/answer/statement examples, not drawn from the above benchmarks.

**Please verify current license terms against each source's own repository/page before redistribution** — third-party license terms can change independently of this repository.

## Summary

- Code in `src/`: see [`LICENSE`](LICENSE) (MIT).
- Original annotations in `data/`: CC BY 4.0 (this file).
- Underlying QA questions/answers in `data/`: licensed by their original source (ARC-Easy, OpenBookQA, MMLU) as noted above.
