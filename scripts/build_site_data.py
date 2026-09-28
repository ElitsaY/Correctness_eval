"""Export the pipeline outputs as JSON for the project website (docs/).

    python scripts/score_baselines.py
    python scripts/evaluate_dataset.py
    python scripts/build_site_data.py

Writes docs/data/:
    scores.json   per-row labels, split, every metric's score and CAP's NLI probabilities
    rows.json     per-row text (question, answers, statements), loaded lazily by the site
    results.json  benchmark tables for the test split + run provenance
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cap_eval import constants as C
from cap_eval.evaluation import ROW_ID, load_correctness_data

logger = logging.getLogger("build_site_data")

NLI_COLUMNS = {
    "p_entailment_forward": "pe_f",
    "p_neutral_forward": "pn_f",
    "p_contradiction_forward": "pc_f",
    "p_entailment_reverse": "pe_r",
    "p_neutral_reverse": "pn_r",
    "p_contradiction_reverse": "pc_r",
}
TEXT_COLUMNS = ["question", "gold_answer", "answer", "premise", "hypothesis"]
# Test-split rows shown on the taxonomy cards, chosen so each class is obvious at a glance.
EXAMPLE_ROWS = {
    "exact": 5588,
    "equivalent": 8778,
    "alternative_correct": 1559,
    "overinclusive_valid": 8107,
    "partial": 3523,
    "overinclusive_invalid": 1844,
    "invalid": 2199,
    "contradictory": 1091,
}
PROBABILITY_DECIMALS = 8  # the in-browser weight explorer matches Table 12 to 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/CAP-Correctness.csv"))
    parser.add_argument("--splits-file", type=Path, default=Path("data/CAP-Correctness-splits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--site-dir", type=Path, default=Path("docs/data"))
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    scores_dir = args.output_dir / "scores"
    eval_dir = args.output_dir / "evaluation" / args.split

    df = load_correctness_data(args.data)
    df = df.merge(pd.read_csv(args.splits_file), on=ROW_ID, validate="one_to_one")
    cap = pd.read_csv(scores_dir / "cap.csv")
    df = df.merge(cap[[ROW_ID, "cap", *NLI_COLUMNS]], on=ROW_ID, validate="one_to_one")
    metrics = ["cap"]
    for metric in C.BASELINE_METRICS:
        path = scores_dir / f"{metric}.csv"
        if path.exists():
            df = df.merge(pd.read_csv(path), on=ROW_ID, validate="one_to_one")
            metrics.append(metric)
    df = df.rename(columns=NLI_COLUMNS)

    args.site_dir.mkdir(parents=True, exist_ok=True)
    scores = {
        "row_id": df[ROW_ID].tolist(),
        "split": df["split"].tolist(),
        "label": df["label"].tolist(),
        "dataset": df["dataset"].tolist(),
        "metrics": metrics,
        # Metric scores keep the precision used for tie detection in the evaluation.
        **{c: _round(df[c], C.SCORE_DECIMALS) for c in metrics},
        **{c: _round(df[c], PROBABILITY_DECIMALS) for c in NLI_COLUMNS.values()},
    }
    _write(args.site_dir / "scores.json", scores)
    _write(args.site_dir / "rows.json", {c: df[c].astype(str).tolist() for c in [ROW_ID, *TEXT_COLUMNS]})

    manifest = json.loads((eval_dir / "evaluate.manifest.json").read_text())
    tables = {
        name: pd.read_csv(eval_dir / f"{name}.csv").replace({np.nan: None}).to_dict(orient="records")
        for name in ("correlations", "descriptives", "class_pairs", "monotonicity", "paired_differences")
    }
    results = {
        "split": args.split,
        "rows": manifest["rows"],
        "n_bootstrap": manifest["settings"]["n_bootstrap"],
        "provenance": {
            "created_utc": manifest["created_utc"],
            "packages": {k: v for k, v in manifest["packages"].items() if v},
        },
        "severity": C.SEVERITY,
        "hard_pairs": [list(p) for p in C.HARD_PAIRS],
        "cap_weights": {"neutral_weight": C.NEUTRAL_WEIGHT, "forward_weight": C.FORWARD_WEIGHT},
        "examples": _examples(df),
        **tables,
    }
    _write(args.site_dir / "results.json", results)


def _examples(df: pd.DataFrame) -> dict[str, dict]:
    """The hand-picked example shown on each taxonomy card."""
    rows = df.set_index(ROW_ID)
    out = {}
    for label, row_id in EXAMPLE_ROWS.items():
        row = rows.loc[row_id]
        if row["label"] != label:
            raise ValueError(f"Example row {row_id} is labeled {row['label']!r}, not {label!r}")
        out[label] = {
            "row_id": row_id,
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "answer": row["answer"],
        }
    return out


def _round(series: pd.Series, decimals: int) -> list[float]:
    return [round(float(x), decimals) for x in series]


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s (%.0f KB)", path, path.stat().st_size / 1024)


if __name__ == "__main__":
    main()
