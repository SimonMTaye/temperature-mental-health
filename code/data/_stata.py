"""Typed Stata IO helpers for eager dataframe reads."""

from pathlib import Path
from typing import Any, cast

import pandas as pd


def read_stata_df(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a Stata file eagerly and type the result as a DataFrame."""
    kwargs.setdefault("iterator", False)
    kwargs.setdefault("chunksize", None)
    return cast(pd.DataFrame, pd.read_stata(path, **kwargs))
