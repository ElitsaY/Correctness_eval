"""Agreement statistics between metric scores and ordinal human labels.

All pairwise statistics are computed exactly in O(n log n) by sorting, so they
can be bootstrapped over the full dataset. Conventions:

* A pair is *comparable* when its two examples have different severity.
* A *win* means the more-correct example has the strictly higher score; score
  ties are counted separately and never as wins.
* ``accuracy = wins / pairs``, ``auc = (wins + 0.5 * ties) / pairs`` and
  ``violation_rate = losses / pairs``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau, spearmanr


@dataclass(frozen=True)
class PairCounts:
    wins: int
    ties: int
    losses: int

    @property
    def pairs(self) -> int:
        return self.wins + self.ties + self.losses

    @property
    def accuracy(self) -> float:
        return self.wins / self.pairs if self.pairs else np.nan

    @property
    def auc(self) -> float:
        return (self.wins + 0.5 * self.ties) / self.pairs if self.pairs else np.nan

    @property
    def violation_rate(self) -> float:
        return self.losses / self.pairs if self.pairs else np.nan

    def __add__(self, other: PairCounts) -> PairCounts:
        return PairCounts(
            self.wins + other.wins, self.ties + other.ties, self.losses + other.losses
        )


ZERO = PairCounts(0, 0, 0)


def compare_groups(better: np.ndarray, worse: np.ndarray) -> PairCounts:
    """Count all (b, w) pairs with b > w, b == w and b < w."""
    if len(better) == 0 or len(worse) == 0:
        return ZERO
    worse_sorted = np.sort(worse)
    below = np.searchsorted(worse_sorted, better, side="left")
    below_or_equal = np.searchsorted(worse_sorted, better, side="right")
    wins = int(below.sum())
    ties = int((below_or_equal - below).sum())
    return PairCounts(wins, ties, len(better) * len(worse) - wins - ties)


def pairwise_ranking(scores: np.ndarray, severity: np.ndarray) -> PairCounts:
    """Compare every pair of examples with different severity."""
    total = ZERO
    lower: list[np.ndarray] = []
    for level in np.unique(severity):  # ascending
        current = scores[severity == level]
        if lower:
            total = total + compare_groups(current, np.concatenate(lower))
        lower.append(current)
    return total


def correlations(scores: np.ndarray, severity: np.ndarray) -> dict[str, float]:
    """Spearman rho and Kendall tau-b between scores and severity."""
    if len(scores) < 3 or np.ptp(scores) == 0 or np.ptp(severity) == 0:
        return {"spearman": np.nan, "spearman_p": np.nan, "kendall": np.nan, "kendall_p": np.nan}
    sp = spearmanr(scores, severity)
    kt = kendalltau(scores, severity)
    return {
        "spearman": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "kendall": float(kt.statistic),
        "kendall_p": float(kt.pvalue),
    }


def bootstrap(
    stat_fn: Callable[[np.ndarray], dict[str, float]],
    n: int,
    n_boot: int,
    seed: int,
    progress: Callable | None = None,
) -> dict[str, np.ndarray]:
    """Nonparametric bootstrap over row indices.

    ``stat_fn`` receives an index array of length ``n`` and returns named
    statistics. Every call with the same ``seed`` and ``n`` draws the same
    resamples, so bootstraps of different metrics are paired and their
    differences can be compared directly.
    """
    rng = np.random.default_rng(seed)
    iterator = range(n_boot) if progress is None else progress(range(n_boot))
    replicates: dict[str, list[float]] = {}
    for _ in iterator:
        idx = rng.integers(0, n, size=n)
        for key, value in stat_fn(idx).items():
            replicates.setdefault(key, []).append(value)
    return {key: np.asarray(values, dtype=float) for key, values in replicates.items()}


def percentile_ci(replicates: np.ndarray, level: float) -> tuple[float, float]:
    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0:
        return np.nan, np.nan
    alpha = (1 - level) / 2 * 100
    low, high = np.percentile(finite, [alpha, 100 - alpha])
    return float(low), float(high)
