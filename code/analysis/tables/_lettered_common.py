"""Shared helpers for the lettered analysis table ports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

PROJECT = Path(__file__).resolve().parents[3]
ANALYSIS_INPUT = PROJECT / "data" / "generated" / "30_analysis_table_input.parquet"
TABLE_DIR = PROJECT / "output" / "tables"

CONTROLS = "age + female + edu_yrs + married + widowed"
CONTROL_COLUMNS = ["age", "female", "edu_yrs", "married", "widowed"]
FE_POOLED = "month + year + wave + kecamatan_code"
FE_IFLS5 = "month + year + kecamatan_code"
FE_WITHIN_DAY = "day_id + kecamatan_code"
CLUSTER = "kabupaten_code"

BASELINE_GROUPS = [
    "palm_farmer_hh_ifls4",
    "coal_worker_hh_ifls4",
    "urban_vehicle_hh_ifls4",
]


def load_analysis() -> pd.DataFrame:
    """Load the canonical analysis input and add table-local indicators."""
    df = pd.read_parquet(ANALYSIS_INPUT).copy()
    require_columns(df, ["wave"])
    df["ifls5"] = (df["wave"] == "IFLS5").astype(int)
    return df


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a compact error for missing analysis columns."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError("analysis input is missing required columns: " + ", ".join(missing))


def nonmissing(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Keep rows with all required values present."""
    required = list(columns)
    require_columns(df, required)
    return df.dropna(subset=required).copy()


def restrict_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with IFLS4 baseline group indicators defined."""
    return nonmissing(df, BASELINE_GROUPS)


def restrict_ifls4_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Keep respondents who appear in IFLS4."""
    require_columns(df, ["pidlink", "wave"])
    baseline_ids = df.loc[df["wave"] == "IFLS4", "pidlink"].dropna().unique()
    return df[df["pidlink"].isin(baseline_ids)].copy()


def drop_singletons(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Drop singleton fixed-effect groups for each listed column."""
    out = df.copy()
    for column in columns:
        require_columns(out, [column])
        counts = out[column].value_counts(dropna=False)
        out = out[out[column].isin(counts[counts > 1].index)].copy()
    return out


def model_frame(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    singleton_columns: Iterable[str] = ("kecamatan_code",),
) -> pd.DataFrame:
    """Create a model frame with required values and identifiable FE groups."""
    out = nonmissing(df, columns)
    out = drop_singletons(out, singleton_columns)
    if out.empty:
        joined = ", ".join(columns)
        raise ValueError(f"model sample is empty after filtering required columns: {joined}")
    return out


def base_required(outcome: str, heat: str = "heat_c_dev") -> list[str]:
    """Common required columns for the main table regressions."""
    return [
        outcome,
        heat,
        CLUSTER,
        "kecamatan_code",
        "month",
        "year",
        "wave",
        *CONTROL_COLUMNS,
    ]


def fit_model(
    df: pd.DataFrame,
    formula: str,
    required_columns: Iterable[str],
    *,
    singleton_columns: Iterable[str] = ("kecamatan_code",),
):
    """Fit a fixed-effect OLS model after table-standard sample filtering."""
    data = model_frame(
        df,
        [*required_columns, CLUSTER],
        singleton_columns=singleton_columns,
    )
    return pf.feols(formula, data=data, vcov={"CRV1": CLUSTER})


def term_stats(model, term: str) -> dict[str, float]:
    """Return coefficient, standard error, p-value, and sample size for one term."""
    return {
        "n": int(model._N),
        "b": float(model.coef().get(term, np.nan)),
        "se": float(model.se().get(term, np.nan)),
        "p": float(model.pvalue().get(term, np.nan)),
    }


def multi_stats(model, terms: Mapping[str, str]) -> dict[str, float]:
    """Return prefixed stats for several terms from one model."""
    row: dict[str, float] = {"n": int(model._N)}
    for prefix, term in terms.items():
        stats = term_stats(model, term)
        row[f"{prefix}_b"] = stats["b"]
        row[f"{prefix}_se"] = stats["se"]
        row[f"{prefix}_p"] = stats["p"]
    return row


def lincom(model, weights: Mapping[str, float]) -> tuple[float, float, float]:
    """Compute a linear combination using the fitted model covariance."""
    coefs = model.coef()
    vcov = pd.DataFrame(model._vcov, index=coefs.index, columns=coefs.index)
    contrast = pd.Series(0.0, index=coefs.index)
    for term, weight in weights.items():
        if term not in contrast.index:
            return np.nan, np.nan, np.nan
        contrast[term] = weight

    estimate = float(contrast @ coefs)
    se = float(np.sqrt(contrast @ vcov @ contrast))
    t_stat = estimate / se if se > 0 else np.nan
    p_value = erfc(abs(t_stat) / sqrt(2)) if not pd.isna(t_stat) else np.nan
    return estimate, se, p_value


def stars(p_value: float) -> str:
    """Return LaTeX significance stars."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return r"^{***}"
    if p_value < 0.05:
        return r"^{**}"
    if p_value < 0.10:
        return r"^{*}"
    return ""


def cell(coef: float, se: float, p_value: float) -> tuple[str, str]:
    """Return a coefficient cell and its standard-error cell."""
    if pd.isna(coef) or pd.isna(se):
        return r"\textemdash", ""
    return f"${coef:+.3f}{stars(p_value)}$", f"$({se:.3f})$"


def fmt_count(value: int | float) -> str:
    """Format counts for LaTeX output."""
    if pd.isna(value):
        return ""
    return f"{int(value):,}"


def write_outputs(
    table_name: str,
    body: list[str],
    rows: Iterable[Mapping[str, object]] | None = None,
) -> None:
    """Write lettered table outputs with matching names."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    body_text = "\n".join(body) + "\n"
    (TABLE_DIR / f"{table_name}_body.tex").write_text(body_text, encoding="utf-8")
    (TABLE_DIR / f"{table_name}.tex").write_text(body_text, encoding="utf-8")
    if rows is not None:
        pd.DataFrame(rows).to_csv(TABLE_DIR / f"{table_name}.csv", index=False)


def add_yes_rows(body: list[str], *, columns: int, sample: str, n_values: list[int]) -> None:
    """Append common controls, FE, sample, and observation rows."""
    body.append(r"\midrule")
    body.append("Demographic controls & " + " & ".join(["Yes"] * columns) + r" \\")
    body.append("Kecamatan FE & " + " & ".join(["Yes"] * columns) + r" \\")
    body.append("Month + Year FE & " + " & ".join(["Yes"] * columns) + r" \\")
    body.append(r"\addlinespace[3pt]")
    body.append("Sample & " + " & ".join([sample] * columns) + r" \\")
    body.append("Observations & " + " & ".join(fmt_count(n) for n in n_values) + r" \\")
