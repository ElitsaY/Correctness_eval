"""Typed experiment configuration loaded from YAML.

Every field is required: the YAML file is the single source of truth for a run,
so an experiment is fully described by its config file plus any ``--set``
overrides (both are recorded in the run manifest).
"""

from __future__ import annotations

import copy
import typing
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is malformed or inconsistent."""


@dataclass(frozen=True)
class DataConfig:
    correctness_csv: str
    statements_csv: str
    label_column: str


@dataclass(frozen=True)
class LabelConfig:
    # Ordinal severity per correctness label: higher = more correct.
    severity: dict[str, int]
    # (better, worse) label pairs that are hard to separate; reported separately.
    hard_pairs: list[list[str]]


@dataclass(frozen=True)
class TrainingConfig:
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    max_grad_norm: float
    early_stopping_patience: int
    metric_for_best_model: str
    save_total_limit: int
    logging_steps: int
    full_determinism: bool


@dataclass(frozen=True)
class GenerationConfig:
    max_input_length: int
    max_new_tokens: int
    num_beams: int
    batch_size: int


@dataclass(frozen=True)
class StatementModelConfig:
    base_model: str
    revision: str
    input_template: str
    max_input_length: int
    max_target_length: int
    holdout_fraction: float
    stratify_column: str
    training: TrainingConfig
    generation: GenerationConfig


@dataclass(frozen=True)
class CapConfig:
    nli_model: str
    revision: str
    batch_size: int
    max_length: int
    fp16: bool
    neutral_weight: float
    forward_weight: float


@dataclass(frozen=True)
class BaselineConfig:
    bertscore_model: str
    bertscore_batch_size: int
    comet_model: str
    comet_batch_size: int
    comet_gpus: int


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: list[str]
    reference_metric: str
    n_bootstrap: int
    ci_level: float


@dataclass(frozen=True)
class Config:
    seed: int
    device: str
    output_dir: str
    data: DataConfig
    labels: LabelConfig
    statement_model: StatementModelConfig
    cap: CapConfig
    baselines: BaselineConfig
    evaluation: EvaluationConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        severity = self.labels.severity
        for pair in self.labels.hard_pairs:
            if len(pair) != 2:
                raise ConfigError(f"labels.hard_pairs entries must be [better, worse], got {pair}")
            better, worse = pair
            for label in pair:
                if label not in severity:
                    raise ConfigError(f"labels.hard_pairs uses unknown label {label!r}")
            if severity[better] <= severity[worse]:
                raise ConfigError(
                    f"labels.hard_pairs: {better!r} must have higher severity than {worse!r}"
                )
        for name in ("neutral_weight", "forward_weight"):
            value = getattr(self.cap, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"cap.{name} must be in [0, 1], got {value}")
        if not 0.0 < self.statement_model.holdout_fraction < 1.0:
            raise ConfigError("statement_model.holdout_fraction must be in (0, 1)")
        if not 0.0 < self.evaluation.ci_level < 1.0:
            raise ConfigError("evaluation.ci_level must be in (0, 1)")
        if self.evaluation.n_bootstrap < 1:
            raise ConfigError("evaluation.n_bootstrap must be positive")
        if self.evaluation.reference_metric not in self.evaluation.metrics:
            raise ConfigError("evaluation.reference_metric must be one of evaluation.metrics")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ConfigError(f"device must be auto/cpu/cuda/mps, got {self.device!r}")


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config and apply ``section.key=value`` overrides."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = apply_overrides(raw, overrides or [])
    config = _build(Config, raw, "config")
    config.validate()
    return config


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Return a copy of ``raw`` with dotted-key overrides applied.

    Values are parsed as YAML, so ``cap.fp16=false`` gives a bool and
    ``evaluation.metrics=[cap,bleu]`` gives a list. Only existing keys can be set.
    """
    raw = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override must look like key.path=value, got {item!r}")
        key_path, value = item.split("=", 1)
        keys = key_path.strip().split(".")
        node = raw
        for key in keys[:-1]:
            if not isinstance(node, dict) or key not in node:
                raise ConfigError(f"Unknown config key in override: {key_path}")
            node = node[key]
        if not isinstance(node, dict) or keys[-1] not in node:
            raise ConfigError(f"Unknown config key in override: {key_path}")
        node[keys[-1]] = yaml.safe_load(value)
    return raw


_SCALARS = (bool, int, float, str)


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    missing = sorted(names - set(data))
    if unknown:
        raise ConfigError(f"{path}: unknown keys {unknown}")
    if missing:
        raise ConfigError(f"{path}: missing keys {missing}")

    kwargs = {}
    for f in fields(cls):
        expected = hints[f.name]
        value = data[f.name]
        key = f"{path}.{f.name}"
        if is_dataclass(expected):
            kwargs[f.name] = _build(expected, value, key)
        elif expected in _SCALARS:
            kwargs[f.name] = _check_scalar(expected, value, key)
        else:
            origin = typing.get_origin(expected)
            if origin is not None and not isinstance(value, origin):
                raise ConfigError(f"{key} must be a {origin.__name__}, got {value!r}")
            kwargs[f.name] = value
    return cls(**kwargs)


def _check_scalar(expected: type, value: Any, key: str) -> Any:
    # bool is a subclass of int, so check it explicitly in both directions.
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, bool) != (expected is bool) or not isinstance(value, expected):
        hint = " (YAML needs a decimal point for floats, e.g. 5.0e-5)" if expected is float else ""
        raise ConfigError(f"{key} must be {expected.__name__}, got {value!r}{hint}")
    return value
