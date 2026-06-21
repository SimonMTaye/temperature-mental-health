"""Table D: summary statistics from the canonical analysis input."""

from __future__ import annotations
from analysis.tables_v2.table_i_economics import ECONOMIC_OUTCOMES, winsorized
from library.config import TABLE_OUTPUT
from library.specs import analysis_df, JOB_LOSS_MAIN

import pandas as pd

TABLE = "table_d_sumstats"

analysis_df["job_earnings_hh_real_usd"] = (
    analysis_df["job_earnings_hh_real"] * analysis_df["conversion_factor"]
)
analysis_df["expenditure_nonfood_total_mo_real_usd"] = (
    analysis_df["expenditure_nonfood_total_mo_real"] * analysis_df["conversion_factor"]
)
analysis_df["expenditure_food_total_mo_real_usd"] = (
    analysis_df["expenditure_food_total_mo_real"] * analysis_df["conversion_factor"]
)


PANELS = [
    (
        "A. Mental-health outcome",
        [
            ("CES-D raw score", "cesd_raw"),
            ("Depressed", "depressed"),
        ],
    ),
    (
        "B. Temperature exposure and variation",
        [
            ("Daily mean temperature", "tmean_c"),
            ("7-day mean temperature", "tmean_7d"),
            ("30-day mean temperature", "tmean_30d"),
            ("7-day mean wet bulb ", "wetbulb_7d"),
        ],
    ),
    (
        "C. Stressors",
        [
            ("Job loss", JOB_LOSS_MAIN),
            ("Palm Farmers", "palm_farmer_hh_ifls4"),
            ("Urban Vehicle Owners", "urban_vehicle_hh_ifls4"),
        ],
    ),
    (
        "D. Economic outcomes",
        [
            ("Monthly Work Income (USD)", "job_earnings_hh_real_usd"),
            (
                "Monthly Nonfood Expenditure (USD)",
                "expenditure_nonfood_total_mo_real_usd",
            ),
            ("Monthly Food Expenditure (USD)", "expenditure_food_total_mo_real_usd"),
            ("Share of Expenditure on Fuel", "fuel_share"),
        ],
    ),
]


def summarize(
    df: pd.DataFrame, panel: str, label: str, variable: str
) -> dict[str, object]:
    values = pd.to_numeric(df[variable], errors="coerce")
    if "real" in variable:
        values = (
            winsorized(values)
            if variable.endswith("_usd")
            else winsorized(values) / 1000
        )
    values = values.dropna()
    return {
        "panel": panel,
        "label": label,
        "var": variable,
        "mean": values.mean(),
        "sd": values.std(),
        "p25": values.quantile(0.25),
        "p50": values.quantile(0.50),
        "p75": values.quantile(0.75),
        "min": values.min(),
        "max": values.max(),
        "n": int(values.size),
    }


def make_table() -> None:
    rows: list[dict[str, object]] = []
    body = [
        # r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Variable & Mean & SD \\",
        r"\midrule",
    ]
    for panel, variables in PANELS:
        body.append(rf"\multicolumn{{3}}{{l}}{{\text{{{panel}}}}} \\ \midrule")
        for label, variable in variables:
            row = summarize(analysis_df, panel, label, variable)
            rows.append(row)
            digits = 3
            if variable in ECONOMIC_OUTCOMES or variable.endswith("_usd"):
                digits = 2
            body.append(
                rf"\quad {label} & {row['mean']:.{digits}f} & {row['sd']:.{digits}f} \\"
            )
        body.append(r"\midrule \addlinespace[1ex]")
    body.extend(
        [
            rf"Kabupaten clusters & & {analysis_df['kabupaten_full_code'].nunique():,}  \\",
            rf"Kecamatan fixed-effect units & & {analysis_df['kecamatan_full_code'].nunique():,}  \\",
            rf"Observations & & {len(analysis_df):,} \\",
            rf"\quad IFLS4 & & {len(analysis_df[~analysis_df['ifls5']]):,}  \\",
            rf"\quad IFLS5 & & {len(analysis_df[analysis_df['ifls5']]):,}  \\",
            r"\bottomrule",
            r"\end{tabular}",
            # r"\end{adjustbox}",
        ]
    )

    body_text = "\n".join(body) + "\n"
    output_path = TABLE_OUTPUT / "table_h_sumstats.tex"
    output_path.write_text(body_text, encoding="utf-8")


if __name__ == "__main__":
    make_table()
