"""Shared analysis-data loading and sample helpers."""

from __future__ import annotations

from pathlib import Path
import sys
from collections.abc import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a compact error if required analysis columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"analysis input is missing required columns: {joined}")


def keep_nonmissing_rows(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Keep only rows with non-missing values in all specified columns."""
    require_columns(df, columns)
    return df.dropna(subset=columns).copy()


def drop_singleton_clusters(
    df: pd.DataFrame,
    cluster: str = "kabupaten_code",
) -> pd.DataFrame:
    """Drop clusters with only one row, matching pyfixest FE table practice."""
    require_columns(df, [cluster])
    counts = df[cluster].value_counts(dropna=False)
    return df[df[cluster].isin(counts[counts > 1].index)].copy()
