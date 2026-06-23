"""Table N: fuel cut mechanism — does the cash-transfer buffer matter?

The headline fuel result is Heat x post_subsidy x urban_vehicle_hh_ifls4 within IFLS5.
This table tests whether the effect concentrates in urban+vehicle HHs that did NOT
have a cash-transfer buffer (PSKS / BLT card) at the time of the Nov 2014 fuel hike.

Cash-transfer buffer at IFLS5 interview:
  has_fuel_buffer = cash_transfer_recipient == 1
  no_fuel_buffer  = 1 - has_fuel_buffer

Specifications (IFLS5 only, kecamatan FE, kabupaten-clustered SE):
  (1) Headline reminder: Heat x urban_veh x post_subsidy
  (2) Quadruple: Heat x urban_veh x post_subsidy x no_fuel_buffer
  (3) Within urban_veh: Heat x post_subsidy x no_fuel_buffer
  (4) Split sample: urban_veh + no_buffer    -> Heat x post_subsidy
  (5) Split sample: urban_veh + with_buffer  -> Heat x post_subsidy
"""

from __future__ import annotations

import pandas as pd

from _lettered_common import (
    CONTROLS,
    FE_IFLS5,
    cell,
    fit_model,
    load_analysis,
    restrict_panel,
    term_stats,
    write_outputs,
)

TABLE = "table_n_fuel_subsidy_card_mechanism"

REQUIRED_BASE = [
    "cesd_z",
    "heat_c_dev",
    "post_subsidy",
    "urban_vehicle_hh_ifls4",
    "no_fuel_buffer",
    "month",
    "year",
    "kecamatan_code",
    "kabupaten_code",
    "age",
    "female",
    "edu_yrs",
    "married",
    "widowed",
]


def build_buffer_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cash_transfer_recipient"] = df["cash_transfer_recipient"].fillna(0).astype(int)
    df["has_fuel_buffer"] = (df["cash_transfer_recipient"] == 1).astype(int)
    df["no_fuel_buffer"] = (1 - df["has_fuel_buffer"]).astype(int)
    return df


def fit_term(df: pd.DataFrame, formula: str, term: str, required: list[str]) -> dict[str, float]:
    model = fit_model(df, formula, required)
    return term_stats(model, term)


def main() -> None:
    df = restrict_panel(load_analysis())
    df = build_buffer_indicators(df)
    sub5 = df[df["wave"] == "IFLS5"].copy()
    uv = sub5[sub5["urban_vehicle_hh_ifls4"] == 1].copy()
    uv_nb = uv[uv["no_fuel_buffer"] == 1].copy()
    uv_b = uv[uv["no_fuel_buffer"] == 0].copy()

    specs = [
        (
            "Heat x post_subsidy x urban_vehicle_hh_ifls4 (headline)",
            sub5,
            f"cesd_z ~ heat_c_dev * post_subsidy * urban_vehicle_hh_ifls4 + {CONTROLS} | {FE_IFLS5}",
            "heat_c_dev:post_subsidy:urban_vehicle_hh_ifls4",
            REQUIRED_BASE,
        ),
        (
            "Heat x post_subsidy x urban_veh x no_fuel_buffer (quadruple)",
            sub5,
            (
                "cesd_z ~ heat_c_dev * post_subsidy * urban_vehicle_hh_ifls4 "
                f"* no_fuel_buffer + {CONTROLS} | {FE_IFLS5}"
            ),
            "heat_c_dev:post_subsidy:urban_vehicle_hh_ifls4:no_fuel_buffer",
            REQUIRED_BASE,
        ),
        (
            "Within urban_veh: Heat x post_subsidy x no_fuel_buffer",
            uv,
            (
                "cesd_z ~ heat_c_dev * post_subsidy * no_fuel_buffer "
                f"+ {CONTROLS} | {FE_IFLS5}"
            ),
            "heat_c_dev:post_subsidy:no_fuel_buffer",
            REQUIRED_BASE,
        ),
        (
            "Within urban_veh + NO buffer: Heat x post_subsidy",
            uv_nb,
            f"cesd_z ~ heat_c_dev * post_subsidy + {CONTROLS} | {FE_IFLS5}",
            "heat_c_dev:post_subsidy",
            [c for c in REQUIRED_BASE if c not in ("no_fuel_buffer", "urban_vehicle_hh_ifls4")],
        ),
        (
            "Within urban_veh + WITH buffer: Heat x post_subsidy",
            uv_b,
            f"cesd_z ~ heat_c_dev * post_subsidy + {CONTROLS} | {FE_IFLS5}",
            "heat_c_dev:post_subsidy",
            [c for c in REQUIRED_BASE if c not in ("no_fuel_buffer", "urban_vehicle_hh_ifls4")],
        ),
    ]

    rows = []
    for label, data, formula, term, required in specs:
        rows.append({"label": label, "term": term, **fit_term(data, formula, term, required)})

    body = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Specification & Key coefficient & N \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Dependent variable: CES-D z-score (IFLS5 only)}} \\",
        r"\addlinespace[2pt]",
    ]
    for row in rows:
        coef_cell, se_cell = cell(row["b"], row["se"], row["p"])
        body.append(rf"{row['label']} & {coef_cell} & {int(row['n']):,} \\")
        body.append(rf" & {se_cell} & \\")
        body.append(r"\addlinespace[2pt]")
    body.extend(
        [
            r"\midrule",
            r"Demographic controls & Yes & \\",
            r"Kecamatan + Month + Year FE & Yes & \\",
            r"Cluster: kabupaten & Yes & \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
