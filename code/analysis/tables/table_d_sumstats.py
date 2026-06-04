"""Table D: summary statistics from the canonical analysis input."""

from __future__ import annotations

import pandas as pd

from _lettered_common import BASELINE_GROUPS, load_analysis, model_frame, require_columns, restrict_panel, write_outputs

TABLE = "table_d_sumstats"

PANELS = [
    (
        "A. Mental-health outcome",
        [("CES-D z-score", "cesd_z"), ("CES-D raw score", "cesd_raw"), ("Depressed", "depressed")],
        "panel",
    ),
    (
        "B. Daily temperature exposure",
        [("Mean temperature", "tmean_c"), ("Heat deviation", "heat_c_dev"), ("Maximum temperature", "tmax_c"), ("Minimum temperature", "tmin_c")],
        "panel",
    ),
    (
        "C. Stressors and baseline groups",
        [
            ("Job loss within 12 months", "job_loss_1_yr"),
            ("Palm farmer household baseline", "palm_farmer_hh_ifls4"),
            ("Coal worker household baseline", "coal_worker_hh_ifls4"),
            ("Urban vehicle household baseline", "urban_vehicle_hh_ifls4"),
        ],
        "panel",
    ),
    (
        "D. Fuel-cut variables",
        [("Post subsidy cut", "post_subsidy"), ("Transport share", "transport_share"), ("Transport spending", "transport_spending_mo")],
        "ifls5",
    ),
    (
        "E. Demographic controls",
        [("Age", "age"), ("Female", "female"), ("Education years", "edu_yrs"), ("Married", "married"), ("Widowed", "widowed")],
        "panel",
    ),
]


def summarize(df: pd.DataFrame, panel: str, label: str, variable: str) -> dict[str, object]:
    values = pd.to_numeric(df[variable], errors="coerce").dropna()
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


def main() -> None:
    df = restrict_panel(load_analysis())
    required = [
        "cesd_z",
        "tmean_c",
        "kecamatan_fe_code",
        "kabupaten_cluster_code",
        "month",
        "year",
        "wave",
        *BASELINE_GROUPS,
    ]
    all_vars = [var for _, rows, _ in PANELS for _, var in rows]
    require_columns(df, [*required, *all_vars])
    panel_df = model_frame(df, required)
    ifls5_df = panel_df[panel_df["wave"] == "IFLS5"].copy()

    rows: list[dict[str, object]] = []
    body = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Variable & Mean & SD & N \\",
        r"\midrule",
    ]
    for panel, variables, sample in PANELS:
        data = ifls5_df if sample == "ifls5" else panel_df
        body.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{panel}}}}} \\")
        for label, variable in variables:
            row = summarize(data, panel, label, variable)
            rows.append(row)
            body.append(rf"\quad {label} & {row['mean']:.3f} & {row['sd']:.3f} & {row['n']:,} \\")
        body.append(r"\addlinespace[3pt]")
    body.extend(
        [
            r"\midrule",
            rf"Kabupaten clusters & \multicolumn{{3}}{{c}}{{{panel_df['kabupaten_cluster_code'].nunique():,}}} \\",
            rf"Kecamatan fixed-effect units & \multicolumn{{3}}{{c}}{{{panel_df['kecamatan_fe_code'].nunique():,}}} \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
