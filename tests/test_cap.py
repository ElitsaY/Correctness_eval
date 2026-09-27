import numpy as np
import pytest

from cap_eval.cap import NLIModel, cap_score, directional_score


def test_directional_score():
    probs = np.array([[0.7, 0.2, 0.1], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(directional_score(probs, 0.3), [0.76, 0.3])


def test_cap_score_weights_directions():
    forward = np.array([[1.0, 0.0, 0.0]])
    reverse = np.array([[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(cap_score(forward, reverse, 0.3, 0.85), [0.85])
    np.testing.assert_allclose(cap_score(reverse, forward, 0.3, 0.85), [0.15])


def test_cap_score_is_bounded():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=(2, 1000))
    scores = cap_score(probs[0], probs[1], 0.3, 0.85)
    assert scores.min() >= 0 and scores.max() <= 1


def test_resolve_class_ids():
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    assert NLIModel._resolve_class_ids(id2label) == [1, 2, 0]
    with pytest.raises(ValueError):
        NLIModel._resolve_class_ids({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
