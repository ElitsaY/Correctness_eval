import numpy as np
import pytest

from cap_eval.stats import PairCounts, bootstrap, compare_groups, pairwise_ranking, percentile_ci


def brute_force_groups(better, worse):
    wins = sum(b > w for b in better for w in worse)
    ties = sum(b == w for b in better for w in worse)
    return PairCounts(wins, ties, len(better) * len(worse) - wins - ties)


def brute_force_ranking(scores, severity):
    total = PairCounts(0, 0, 0)
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if severity[i] == severity[j]:
                continue
            hi, lo = (i, j) if severity[i] > severity[j] else (j, i)
            total = total + brute_force_groups([scores[hi]], [scores[lo]])
    return total


@pytest.mark.parametrize("seed", range(5))
def test_compare_groups_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    # Rounded values so that ties occur.
    better = rng.normal(size=40).round(1)
    worse = rng.normal(size=30).round(1)
    assert compare_groups(better, worse) == brute_force_groups(better, worse)


@pytest.mark.parametrize("seed", range(5))
def test_pairwise_ranking_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    severity = rng.integers(0, 5, size=120)
    scores = (severity + rng.normal(scale=2, size=120)).round(0)
    assert pairwise_ranking(scores, severity) == brute_force_ranking(scores, severity)


def test_pair_counts_rates():
    counts = PairCounts(wins=6, ties=2, losses=2)
    assert counts.pairs == 10
    assert counts.accuracy == pytest.approx(0.6)
    assert counts.auc == pytest.approx(0.7)
    assert counts.violation_rate == pytest.approx(0.2)
    assert np.isnan(PairCounts(0, 0, 0).accuracy)


def test_compare_groups_empty():
    assert compare_groups(np.array([]), np.array([1.0])).pairs == 0


def test_bootstrap_is_reproducible_and_paired():
    values = np.arange(50, dtype=float)
    seen = []

    def record(idx):
        seen.append(idx.copy())
        return {"mean": values[idx].mean()}

    first = bootstrap(record, n=50, n_boot=20, seed=7)
    first_idx = list(seen)
    seen.clear()
    second = bootstrap(record, n=50, n_boot=20, seed=7)

    np.testing.assert_array_equal(first["mean"], second["mean"])
    for a, b in zip(first_idx, seen):
        np.testing.assert_array_equal(a, b)


def test_percentile_ci_ignores_nan():
    low, high = percentile_ci(np.array([np.nan, *range(101)], dtype=float), level=0.9)
    assert (low, high) == pytest.approx((5.0, 95.0))
