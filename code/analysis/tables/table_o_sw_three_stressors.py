"""Table O: subjective wellbeing channel for palm, fuel, and job loss.

Reads b3a_sw items at the individual level and tests:
  Palm:     SW ~ palm_shock + palm_farmer_hh_ifls4 + ... | month+year+wave+kec   (panel)
            where palm_shock = palm_farmer_hh_ifls4 * ifls5
  Fuel:     SW ~ fuel_shock + urban_vehicle_hh_ifls4 + ... | month+year+kec      (IFLS5)
            where fuel_shock = urban_vehicle_hh_ifls4 * post_subsidy
  Job loss: SW ~ job_loss_1_yr + ... | month+year+wave+kec                       (panel)

Outcomes (sw01, sw04, sw05, sw06, sw03b, sw12, sw03a) are NOT in the canonical
analysis input — sidecar read from raw IFLS b3a_sw with codes 8 (don't know) and
9 (missing) set to NaN. Raw IFLS folders come from code/data/config.py.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from _lettered_common import (
    CONTROLS,
    FE_IFLS5,
    FE_POOLED,
    PROJECT,
    cell,
    fit_model,
    load_analysis,
    restrict_panel,
    term_stats,
    write_outputs,
)

# Raw IFLS sidecar folders live in the data pipeline's central path config.
sys.path.insert(0, str(PROJECT / "code" / "data"))
from config import IFLS4_FOLDER, IFLS5_FOLDER  # noqa: E402

TABLE = "table_o_sw_three_stressors"

SW_FILES = {
    "IFLS4": IFLS4_FOLDER / "b3a_sw.dta",
    "IFLS5": IFLS5_FOLDER / "b3a_sw.dta",
}

SW_OUT = ["sw01", "sw04", "sw05", "sw06", "sw03b", "sw12", "sw03a"]
SW_LABELS = {
    "sw01":  "Cantril ladder (1-6, higher=richer)",
    "sw04":  "Living std adequacy (1-3)",
    "sw05":  "Food adequacy (1-3)",
    "sw06":  "Health adequacy (1-3)",
    "sw03b": "Family life adequacy (1-3)",
    "sw12":  "Overall happy (1=happy, 4=unhappy)",
    "sw03a": "Keep std 5yr (1=v.likely, 4=v.unlikely)",
}


def load_sw() -> pd.DataFrame:
    frames = []
    for wave, path in SW_FILES.items():
        sw = pd.read_stata(path, convert_categoricals=False, columns=["pidlink", *SW_OUT])
        sw["wave"] = wave
        for col in SW_OUT:
            sw[col] = pd.to_numeric(sw[col], errors="coerce")
            sw.loc[sw[col].isin([8, 9]), col] = np.nan
        frames.append(sw)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["pidlink", "wave"])


def fit_palm(df: pd.DataFrame, outcome: str) -> dict[str, float]:
    formula = (
        f"{outcome} ~ palm_shock + palm_farmer_hh_ifls4 + {CONTROLS} | {FE_POOLED}"
    )
    required = [
        outcome, "palm_shock", "palm_farmer_hh_ifls4",
        "month", "year", "wave", "kecamatan_code", "kabupaten_code",
        "age", "female", "edu_yrs", "married", "widowed",
    ]
    model = fit_model(df, formula, required)
    return term_stats(model, "palm_shock")


def fit_fuel(df: pd.DataFrame, outcome: str) -> dict[str, float]:
    formula = (
        f"{outcome} ~ fuel_shock + urban_vehicle_hh_ifls4 + {CONTROLS} | {FE_IFLS5}"
    )
    required = [
        outcome, "fuel_shock", "urban_vehicle_hh_ifls4",
        "month", "year", "kecamatan_code", "kabupaten_code",
        "age", "female", "edu_yrs", "married", "widowed",
    ]
    model = fit_model(df, formula, required)
    return term_stats(model, "fuel_shock")


def fit_jobloss(df: pd.DataFrame, outcome: str, treatment: str) -> dict[str, float]:
    formula = f"{outcome} ~ {treatment} + {CONTROLS} | {FE_POOLED}"
    required = [
        outcome, treatment,
        "month", "year", "wave", "kecamatan_code", "kabupaten_code",
        "age", "female", "edu_yrs", "married", "widowed",
    ]
    model = fit_model(df, formula, required)
    return term_stats(model, treatment)


def main() -> None:
    df = restrict_panel(load_analysis())
    df = df.merge(load_sw(), on=["pidlink", "wave"], how="left")
    df["palm_shock"] = df["palm_farmer_hh_ifls4"].fillna(0).astype(int) * df["ifls5"]
    df["fuel_shock"] = (
        df["post_subsidy"].fillna(0).astype(int) * df["urban_vehicle_hh_ifls4"].fillna(0).astype(int)
    )
    sub5 = df[df["wave"] == "IFLS5"].copy()

    rows = []
    for outcome in SW_OUT:
        palm_stats = fit_palm(df, outcome)
        fuel_stats = fit_fuel(sub5, outcome)
        jl_any = fit_jobloss(df, outcome, "job_loss_1_yr")
        rows.append({
            "outcome": outcome,
            "label": SW_LABELS[outcome],
            "palm_b": palm_stats["b"], "palm_se": palm_stats["se"], "palm_p": palm_stats["p"],
            "palm_n": palm_stats["n"],
            "fuel_b": fuel_stats["b"], "fuel_se": fuel_stats["se"], "fuel_p": fuel_stats["p"],
            "fuel_n": fuel_stats["n"],
            "jl_any_b": jl_any["b"], "jl_any_se": jl_any["se"], "jl_any_p": jl_any["p"],
            "jl_any_n": jl_any["n"],
        })

    body = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & (1) & (2) & (3) \\",
        r" & Palm $\times$ IFLS5 & UrbanVeh $\times$ PostFuel & Job loss \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Dependent variables: subjective wellbeing items}} \\",
        r"\addlinespace[2pt]",
    ]
    for row in rows:
        cells = []
        ses = []
        for prefix in ("palm", "fuel", "jl_any"):
            coef_cell, se_cell = cell(
                row[f"{prefix}_b"], row[f"{prefix}_se"], row[f"{prefix}_p"]
            )
            cells.append(coef_cell)
            ses.append(se_cell)
        body.append(rf"{row['label']} & " + " & ".join(cells) + r" \\")
        body.append(r" & " + " & ".join(ses) + r" \\")
        body.append(r"\addlinespace[2pt]")
    body.append(r"\midrule")
    n_palm = rows[0]["palm_n"]
    n_fuel = rows[0]["fuel_n"]
    n_jl   = rows[0]["jl_any_n"]
    body.extend(
        [
            r"Demographic controls & Yes & Yes & Yes \\",
            r"Kecamatan FE & Yes & Yes & Yes \\",
            r"Month + Year FE & Yes & Yes & Yes \\",
            r"Wave FE & Yes & --- & Yes \\",
            r"\addlinespace[3pt]",
            r"Sample & Panel & IFLS5 panel & Panel \\",
            (
                "Observations (sw01) & "
                + " & ".join(f"{int(v):,}" for v in [n_palm, n_fuel, n_jl])
                + r" \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
