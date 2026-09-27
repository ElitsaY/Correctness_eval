from pathlib import Path

import pytest
import yaml

from cap_eval.config import ConfigError, apply_overrides, load_config

DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_config_loads():
    config = load_config(DEFAULT)
    assert config.statement_model.training.learning_rate == pytest.approx(5e-5)
    assert config.evaluation.reference_metric in config.evaluation.metrics


def test_overrides_are_parsed_as_yaml():
    config = load_config(DEFAULT, ["cap.fp16=false", "evaluation.metrics=[cap, bleu]"])
    assert config.cap.fp16 is False
    assert config.evaluation.metrics == ["cap", "bleu"]


def test_unknown_override_key_is_rejected():
    with pytest.raises(ConfigError):
        apply_overrides({"cap": {"batch_size": 1}}, ["cap.batch_sise=2"])


def _write(tmp_path, raw):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_unknown_and_missing_keys_are_rejected(tmp_path):
    raw = yaml.safe_load(DEFAULT.read_text())
    raw["cap"]["typo"] = 1
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(_write(tmp_path, raw))
    raw = yaml.safe_load(DEFAULT.read_text())
    del raw["seed"]
    with pytest.raises(ConfigError, match="missing keys"):
        load_config(_write(tmp_path, raw))


def test_string_float_is_rejected(tmp_path):
    raw = yaml.safe_load(DEFAULT.read_text())
    raw["statement_model"]["training"]["learning_rate"] = "5e-5"
    with pytest.raises(ConfigError, match="decimal point"):
        load_config(_write(tmp_path, raw))


def test_hard_pairs_must_be_ordered(tmp_path):
    raw = yaml.safe_load(DEFAULT.read_text())
    raw["labels"]["hard_pairs"] = [["partial", "equivalent"]]
    with pytest.raises(ConfigError, match="higher severity"):
        load_config(_write(tmp_path, raw))
