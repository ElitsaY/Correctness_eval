"""Benchmark metric scores against ordinal human correctness labels."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import EvaluationConfig, LabelConfig
from .data import LABEL, SEVERITY
from .stats import bootstrap, compare_groups, correlations, pairwise_ranking, percentile_ci

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    tables: dict[str, pd.DataFrame]
    replicates: dict[str, dict[str, np.ndarray]]


def class_pairs(labels: LabelConfig, present: set[str]) -> list[tuple[str, str]]:
    """All (better, worse) label pairs with different severity, most correct first."""
    severity = labels.severity
    ordered = sorted((l for l in severity if l in present), key=lambda l: -severity[l])
    return [(a, b) for a, b in combinations(ordered, 2) if severity[a] > severity[b]]


def evaluate(
    df: pd.DataFrame, labels: LabelConfig, config: EvaluationConfig, seed: int
) -> EvaluationResult:
    """Compute every table for every metric column listed in ``config.metrics``.

    All metrics share the same rows and the same bootstrap resamples.
    """
    for metric in config.metrics:
        if not np.isfinite(df[metric].to_numpy(dtype=float)).all():
            raise ValueError(f"Metric {metric!r} has missing or non-finite scores")

    label_names = [l for l in labels.severity if l in set(df[LABEL])]
    label_codes = pd.Categorical(df[LABEL], categories=label_names).codes
    severity = df[SEVERITY].to_numpy()
    pairs = class_pairs(labels, set(label_names))
    hard = {tuple(p) for p in labels.hard_pairs}
    level = config.ci_level

    def stats_for(scores: np.ndarray, idx: np.ndarray | None = None) -> dict[str, float]:
        s, sev, codes = (scores, severity, label_codes) if idx is None else (
            scores[idx], severity[idx], label_codes[idx]
        )
        out = {k: v for k, v in correlations(s, sev).items() if not k.endswith("_p")}
        out["pairwise_accuracy"] = pairwise_ranking(s, sev).accuracy
        by_label = [s[codes == i] for i in range(len(label_names))]
        for name, values in zip(label_names, by_label):
            out[f"mean:{name}"] = values.mean() if values.size else np.nan
        for better, worse in pairs:
            counts = compare_groups(
                by_label[label_names.index(better)], by_label[label_names.index(worse)]
            )
            out[f"acc:{better}>{worse}"] = counts.accuracy
            out[f"auc:{better}>{worse}"] = counts.auc
        return out

    replicates = {}
    rows: dict[str, list[dict]] = {k: [] for k in ("correlations", "descriptives", "class_pairs")}
    for metric in config.metrics:
        scores = df[metric].to_numpy(dtype=float)
        logger.info("Bootstrapping %s (%d resamples)", metric, config.n_bootstrap)
        boot = bootstrap(
            lambda idx: stats_for(scores, idx),
            n=len(scores),
            n_boot=config.n_bootstrap,
            seed=seed,
            progress=lambda it, m=metric: tqdm(it, desc=f"Bootstrap {metric}", leave=False),
        )
        replicates[metric] = boot

        def ci(key: str) -> dict[str, float]:
            low, high = percentile_ci(boot[key], level)
            return {f"{key}_ci_low": low, f"{key}_ci_high": high}

        corr = correlations(scores, severity)
        ranking = pairwise_ranking(scores, severity)
        rows["correlations"].append({
            "metric": metric,
            "n": len(scores),
            "spearman": corr["spearman"], **ci("spearman"), "spearman_p": corr["spearman_p"],
            "kendall": corr["kendall"], **ci("kendall"), "kendall_p": corr["kendall_p"],
            "pairwise_accuracy": ranking.accuracy, **ci("pairwise_accuracy"),
            "comparable_pairs": ranking.pairs,
            "pair_wins": ranking.wins,
            "pair_ties": ranking.ties,
            "pair_losses": ranking.losses,
        })

        for name in label_names:
            values = scores[df[LABEL].to_numpy() == name]
            low, high = percentile_ci(boot[f"mean:{name}"], level)
            rows["descriptives"].append({
                "metric": metric,
                "label": name,
                "severity": labels.severity[name],
                "n": values.size,
                "mean": values.mean(),
                "mean_ci_low": low,
                "mean_ci_high": high,
                "std": values.std(ddof=1) if values.size > 1 else np.nan,
                "min": values.min(),
                "p05": np.percentile(values, 5),
                "median": np.median(values),
                "p95": np.percentile(values, 95),
                "max": values.max(),
            })

        for better, worse in pairs:
            b = scores[df[LABEL].to_numpy() == better]
            w = scores[df[LABEL].to_numpy() == worse]
            counts = compare_groups(b, w)
            key = f"{better}>{worse}"
            acc_low, acc_high = percentile_ci(boot[f"acc:{key}"], level)
            auc_low, auc_high = percentile_ci(boot[f"auc:{key}"], level)
            rows["class_pairs"].append({
                "metric": metric,
                "better": better,
                "worse": worse,
                "hard_pair": (better, worse) in hard,
                "n_better": b.size,
                "n_worse": w.size,
                "pairs": counts.pairs,
                "wins": counts.wins,
                "ties": counts.ties,
                "losses": counts.losses,
                "accuracy": counts.accuracy,
                "accuracy_ci_low": acc_low,
                "accuracy_ci_high": acc_high,
                "auc": counts.auc,
                "auc_ci_low": auc_low,
                "auc_ci_high": auc_high,
                "violation_rate": counts.violation_rate,
                "mean_better": b.mean(),
                "mean_worse": w.mean(),
                "mean_violation": bool(w.mean() > b.mean()),
            })

    class_pair_df = pd.DataFrame(rows["class_pairs"])
    tables = {
        "correlations": pd.DataFrame(rows["correlations"]),
        "descriptives": pd.DataFrame(rows["descriptives"]),
        "class_pairs": class_pair_df,
        "hard_pairs": class_pair_df[class_pair_df["hard_pair"]].reset_index(drop=True),
        "monotonicity": _monotonicity_summary(class_pair_df),
        "paired_differences": _paired_differences(replicates, config, level),
    }
    return EvaluationResult(tables=tables, replicates=replicates)


def _monotonicity_summary(class_pair_df: pd.DataFrame) -> pd.DataFrame:
    """Share of class pairs whose mean scores are in the wrong order."""
    grouped = class_pair_df.groupby("metric", sort=False)
    sums = grouped[["losses", "pairs"]].sum()
    return pd.DataFrame({
        "class_pairs": grouped.size(),
        "mean_violations": grouped["mean_violation"].sum(),
        "mean_violation_rate": grouped["mean_violation"].mean(),
        "row_level_violation_rate": sums["losses"] / sums["pairs"],
    }).reset_index()


def _paired_differences(
    replicates: dict[str, dict[str, np.ndarray]], config: EvaluationConfig, level: float
) -> pd.DataFrame:
    """Reference metric minus each other metric, from the paired bootstrap."""
    ref = config.reference_metric
    rows = []
    for metric in config.metrics:
        if metric == ref:
            continue
        for stat in ("spearman", "kendall", "pairwise_accuracy"):
            diff = replicates[ref][stat] - replicates[metric][stat]
            diff = diff[np.isfinite(diff)]
            low, high = percentile_ci(diff, level)
            rows.append({
                "reference": ref,
                "metric": metric,
                "statistic": stat,
                "mean_difference": diff.mean(),
                "ci_low": low,
                "ci_high": high,
                # One-sided bootstrap p-value for "reference is not better".
                "p_reference_not_better": float((diff <= 0).mean()),
            })
    return pd.DataFrame(rows)


def write_tables(result: EvaluationResult, output_dir: Path, ci_level: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "summary.md").write_text(_summary_markdown(result.tables, ci_level))
    logger.info("Wrote evaluation tables to %s", output_dir)


def _summary_markdown(tables: dict[str, pd.DataFrame], ci_level: float) -> str:
    pct = f"{ci_level:.0%}"

    def with_ci(row: pd.Series, key: str) -> str:
        return f"{row[key]:.4f} [{row[key + '_ci_low']:.4f}, {row[key + '_ci_high']:.4f}]"

    corr = tables["correlations"].sort_values("spearman", ascending=False)
    lines = [
        "# Evaluation summary",
        "",
        f"Point estimates with {pct} bootstrap confidence intervals.",
        "",
        "## Agreement with human severity",
        "",
        _md_table(
            ["Metric", "Spearman", "Kendall tau-b", "Pairwise accuracy"],
            [[r.metric, with_ci(r, "spearman"), with_ci(r, "kendall"), with_ci(r, "pairwise_accuracy")]
             for _, r in corr.iterrows()],
        ),
        "",
        "## Hard pairs (accuracy: better class scored strictly higher)",
        "",
        _md_table(
            ["Metric", "Pair", "Accuracy", "AUC"],
            [[r.metric, f"{r.better} > {r.worse}", with_ci(r, "accuracy"), with_ci(r, "auc")]
             for _, r in tables["hard_pairs"].iterrows()],
        ),
        "",
        "## Monotonicity",
        "",
        _md_table(
            ["Metric", "Class pairs with mean in wrong order", "Row-level violation rate"],
            [[r.metric, f"{r.mean_violations}/{r.class_pairs}", f"{r.row_level_violation_rate:.4f}"]
             for _, r in tables["monotonicity"].iterrows()],
        ),
    ]
    diffs = tables["paired_differences"]
    if not diffs.empty:
        lines += [
            "",
            f"## Paired differences ({diffs['reference'].iloc[0]} minus metric)",
            "",
            _md_table(
                ["Metric", "Statistic", "Difference", "p (reference not better)"],
                [[r.metric, r.statistic, f"{r.mean_difference:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]",
                  f"{r.p_reference_not_better:.4f}"] for _, r in diffs.iterrows()],
            ),
        ]
    return "\n".join(lines) + "\n"


def _md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)
