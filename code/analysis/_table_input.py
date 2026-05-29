"""Shared loader for the table-ready analysis input."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
GENERATED = PROJECT / "data" / "generated"
TABLE_INPUT = GENERATED / "analysis_table_input.parquet"


def load_table_input(*, require_singleton_kab: bool = False) -> pd.DataFrame:
    """Load the curated person-wave analysis input and optionally drop singleton kabs."""
    df = pd.read_parquet(TABLE_INPUT)
    if require_singleton_kab:
        counts = df.kabupaten_code.value_counts()
        df = df[df.kabupaten_code.isin(counts[counts > 1].index)].copy()
    return df
