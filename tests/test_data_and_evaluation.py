import numpy as np
import pandas as pd
import pytest

from cap_eval.config import EvaluationConfig, LabelConfig
from cap_eval.data import ROW_ID, load_correctness_data
from cap_eval.evaluation import class_pairs, evaluate
from cap_eval.statements import format_inputs

SEVERITY = {"exact": 2, "equivalent": 2, "partial": 1, "invalid": 0}
LABELS = LabelConfig(severity=SEVERITY, hard_pairs=[["equivalent", "partial"]])


def _frame(labels, answers=None):
    n = len(labels)
    return pd.DataFrame({
        "question": [f"q{i}" for i in range(n)],
        "gold_answer": ["gold"] * n,
        "answer": answers or ["ans"] * n,
        "premise": ["p"] * n,
        "hypothesis": ["h"] * n,
        "label_semantic": labels,
    })


def test_load_correctness_data_drops_incomplete_rows_and_keeps_row_ids(tmp_path):
    path = tmp_path / "data.csv"
    _frame([" Exact", "partial", "invalid"], answers=["a", None, "c"]).to_csv(path, index=False)
    df = load_correctness_data(str(path), "label_semantic", SEVERITY)
    assert df[ROW_ID].tolist() == [0, 2]
    assert df["label"].tolist() == ["exact", "invalid"]
    assert df["severity"].tolist() == [2, 0]


def test_unknown_label_raises(tmp_path):
    path = tmp_path / "data.csv"
    _frame(["exact", "mystery"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="mystery"):
        load_correctness_data(str(path), "label_semantic", SEVERITY)


def test_class_pairs_skip_equal_severity():
    pairs = class_pairs(LABELS, set(SEVERITY))
    assert ("exact", "equivalent") not in pairs
    assert ("exact", "partial") in pairs
    assert all(SEVERITY[a] > SEVERITY[b] for a, b in pairs)


def test_format_inputs_normalizes_whitespace():
    out = format_inputs(pd.Series(["What  is\nit?"]), pd.Series([" x "]), "question: {question} answer: {answer}")
    assert out == ["question: What is it? answer: x"]


def test_evaluate_is_deterministic_and_perfect_metric_scores_one():
    rng = np.random.default_rng(0)
    labels = rng.choice(list(SEVERITY), size=200)
    df = pd.DataFrame({"label": labels, "severity": [SEVERITY[l] for l in labels]})
    df["perfect"] = df["severity"] + rng.uniform(0, 0.5, size=200)
    df["noisy"] = rng.uniform(size=200)
    config = EvaluationConfig(
        metrics=["perfect", "noisy"], reference_metric="perfect", n_bootstrap=50, ci_level=0.95
    )

    first = evaluate(df, LABELS, config, seed=1)
    second = evaluate(df, LABELS, config, seed=1)
    for name, table in first.tables.items():
        pd.testing.assert_frame_equal(table, second.tables[name])

    corr = first.tables["correlations"].set_index("metric")
    assert corr.loc["perfect", "pairwise_accuracy"] == 1.0
    assert corr.loc["perfect", "pairwise_accuracy_ci_low"] == 1.0
    assert first.tables["monotonicity"].set_index("metric").loc["perfect", "mean_violations"] == 0
    diffs = first.tables["paired_differences"]
    assert (diffs["mean_difference"] > 0).all()
