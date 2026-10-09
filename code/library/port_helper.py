"""Export final Python regression inputs for the explicit Stata port."""
import hashlib
import json
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_string_dtype

from library.config import PROJECT

OUTPUT_DATA = True
EXPORT_DIR = PROJECT / "data" / "generated" / "stata_port"


def save_data(table_code_name: str, regression_id: int, dataset: pd.DataFrame):
    """Save a copy, preserving rows; estimation sample selection stays with engines."""
    if not OUTPUT_DATA:
        return
    data = dataset.copy(deep=True)
    names = {}
    used = set()
    for original in data.columns:
        candidate = re.sub(r"[^A-Za-z0-9_]", "_", original)
        if not candidate or not re.match(r"[A-Za-z_]", candidate):
            candidate = "v_" + candidate
        if len(candidate) > 32 or candidate != original or candidate in used:
            candidate = candidate[:23] + "_" + hashlib.sha256(original.encode()).hexdigest()[:8]
        if candidate in used:
            raise ValueError(f"Stata column-name collision: {original}")
        used.add(candidate)
        names[original] = candidate
    data = data.rename(columns=names)
    # Numeric aliases are only for identifiers, never regression covariates.
    aliases = {}
    data["stata_source_row"] = np.arange(len(dataset), dtype=np.float64)
    for source, alias in (("pidlink", "stata_pidlink"), ("gadm_fullcode", "stata_gadm_fullcode")):
        if source in dataset:
            codes, levels = pd.factorize(dataset[source], sort=True)
            data[alias] = np.where(codes == -1, np.nan, codes + 1).astype(float)
            aliases[alias] = {"source": source, "values": {str(i + 1): str(value) for i, value in enumerate(levels)}}
    if {"month", "year"}.issubset(dataset.columns):
        data["stata_month_times_year"] = dataset["month"] * dataset["year"]
        aliases["stata_month_times_year"] = {"expression": "month * year", "meaning": "Numeric product categorized by pyfixest fixed-effect formula; not month-year interaction"}
    string_missing = []
    date_columns = {}
    for column in data:
        series = data[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            # Preserve category values, rather than silently substituting codes.
            series = series.astype(float) if is_numeric_dtype(series.cat.categories.dtype) else series.astype(object)
            data[column] = series
        if is_bool_dtype(series.dtype):
            data[column] = series.to_numpy(dtype=np.float64, na_value=np.nan)
        elif is_numeric_dtype(series.dtype):
            if isinstance(series.dtype, pd.api.extensions.ExtensionDtype):
                values = series.to_numpy(dtype=np.float64, na_value=np.nan)
                if pd.api.types.is_integer_dtype(series.dtype):
                    observed = series.dropna()
                    if any(int(value) != int(float(value)) for value in observed):
                        raise ValueError(f"Integer precision loss: {column}")
                data[column] = values
            elif series.dtype.kind in "iu" and series.dtype.itemsize >= 8:
                values = series.to_numpy(dtype=np.float64)
                if not np.array_equal(values.astype(series.dtype), series.to_numpy()):
                    raise ValueError(f"Integer precision loss: {column}")
                data[column] = values
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            date_columns[column] = "tc"
        elif is_string_dtype(series.dtype) or series.dtype == object:
            if not series.dropna().map(lambda value: isinstance(value, str)).all():
                raise TypeError(f"Unsupported mixed object values in {column}")
            if series.isna().any():
                string_missing.append(column)
            data[column] = series.fillna("").astype(object)
        else:
            raise TypeError(f"Unsupported export dtype: {column}: {series.dtype}")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{table_code_name}_{regression_id}"
    data.to_stata(EXPORT_DIR / f"{stem}.dta", write_index=False, version=118, convert_dates=date_columns)
    (EXPORT_DIR / f"{stem}_mapping.json").write_text(json.dumps({"row_count": len(data), "column_names": names, "identifier_aliases": aliases, "string_missing_as_empty": string_missing}, indent=2) + "\n")
