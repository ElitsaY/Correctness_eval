"""Loading and validating the CAP datasets."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

import pandas as pd

logger = logging.getLogger(__name__)

ROW_ID = "row_id"
LABEL = "label"
SEVERITY = "severity"

# Columns every metric needs; rows missing any of these are dropped so that all
# metrics are evaluated on exactly the same examples.
CORRECTNESS_TEXT_COLUMNS = ("question", "gold_answer", "answer")
STATEMENT_COLUMNS = ("premise", "hypothesis")


def normalize_text(series: pd.Series) -> pd.Series:
    """Collapse whitespace; used identically for training and generation inputs."""
    return series.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def clean_labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def load_correctness_data(
    path: str,
    label_column: str,
    severity: Mapping[str, int],
    require_statements: bool = True,
) -> pd.DataFrame:
    """Load CAP-Correctness with a stable ``row_id`` and ordinal severity.

    ``row_id`` is the 0-based row position in the original CSV (kept if the file
    already has one, e.g. regenerated statements), so score files produced by
    different commands can always be joined back to the data.
    """
    df = pd.read_csv(path)
    required = [*CORRECTNESS_TEXT_COLUMNS, label_column]
    if require_statements:
        required += STATEMENT_COLUMNS
    _require_columns(df, required, path)

    if ROW_ID not in df.columns:
        df.insert(0, ROW_ID, range(len(df)))
    if not df[ROW_ID].is_unique:
        raise ValueError(f"{path}: {ROW_ID} values must be unique")
    incomplete = df[required].isna().any(axis=1) | (
        df[required].astype(str).apply(lambda col: col.str.strip() == "").any(axis=1)
    )
    if incomplete.any():
        logger.warning(
            "Dropping %d row(s) with missing %s (row_id: %s)",
            int(incomplete.sum()),
            required,
            df.loc[incomplete, ROW_ID].tolist(),
        )
        df = df.loc[~incomplete].copy()

    df[LABEL] = clean_labels(df[label_column])
    unknown = sorted(set(df[LABEL]) - set(severity))
    if unknown:
        raise ValueError(f"Labels not in labels.severity: {unknown}")
    df[SEVERITY] = df[LABEL].map(severity).astype(int)
    return df.reset_index(drop=True)


def load_statement_training_data(path: str) -> pd.DataFrame:
    """Load CAP-Statements (question, answer -> statement) with normalized text."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    columns = ["question", "answer", "statement"]
    _require_columns(df, columns, path)

    df = df.dropna(subset=columns).copy()
    for col in columns:
        df[col] = normalize_text(df[col])
    df = df[(df[columns] != "").all(axis=1)]
    if "language" in df.columns:
        df["language"] = df["language"].astype(str).str.strip()
    return df.reset_index(drop=True)


def _require_columns(df: pd.DataFrame, columns: Iterable[str], path: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
