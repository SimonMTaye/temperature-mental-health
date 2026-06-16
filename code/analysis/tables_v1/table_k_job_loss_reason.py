"""Table K: job-loss reason heterogeneity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from _lettered_common import CONTROLS, FE_POOLED, cell, fit_model, lincom, load_analysis, restrict_ifls4_panel, term_stats, write_outputs

TABLE = "table_k_job_loss_reason"


def fit_reason(df, variable: str) -> dict[str, object]:
    formula = f"cesd_z ~ heat_c_dev * {variable} + {CONTROLS} | {FE_POOLED}"
    required = ["cesd_z", "heat_c_dev", variable, "month", "year", "wave", "kecamatan_fe_code", "age", "female", "edu_yrs", "married", "widowed"]
    model = fit_model(df, formula, required)
    frame = df.dropna(subset=required)
    stats = term_stats(model, f"heat_c_dev:{variable}")
    treated = frame[frame[variable] == 1]
    return {
        **stats,
        "n_treated": int(frame[variable].sum()),
        "female_pct": float(100 * treated["female"].mean()) if not treated.empty else np.nan,
        "stress_b": float(model.coef().get(variable, np.nan)),
        "stress_se": float(model.se().get(variable, np.nan)),
        "stress_p": float(model.pvalue().get(variable, np.nan)),
    }


def fit_joint(df) -> dict[str, object]:
    formula = f"cesd_z ~ heat_c_dev * (involuntary_loss_1_yr + family_loss_1_yr) + {CONTROLS} | {FE_POOLED}"
    required = ["cesd_z", "heat_c_dev", "involuntary_loss_1_yr", "family_loss_1_yr", "month", "year", "wave", "kecamatan_fe_code", "age", "female", "edu_yrs", "married", "widowed"]
    model = fit_model(df, formula, required)
    invol = term_stats(model, "heat_c_dev:involuntary_loss_1_yr")
    family = term_stats(model, "heat_c_dev:family_loss_1_yr")
    diff_b, diff_se, diff_p = lincom(
        model,
        {
            "heat_c_dev:involuntary_loss_1_yr": 1.0,
            "heat_c_dev:family_loss_1_yr": -1.0,
        },
    )
    return {
        "n": int(model._N),
        "invol_b": invol["b"],
        "invol_se": invol["se"],
        "invol_p": invol["p"],
        "family_b": family["b"],
        "family_se": family["se"],
        "family_p": family["p"],
        "diff_b": diff_b,
        "diff_se": diff_se,
        "diff_p": diff_p,
    }


def main() -> None:
    df = restrict_ifls4_panel(load_analysis())
    rows = [
        {"row": "Any job loss", "var": "job_loss_1_yr", **fit_reason(df, "job_loss_1_yr")},
        {"row": "Involuntary", "var": "involuntary_loss_1_yr", **fit_reason(df, "involuntary_loss_1_yr")},
        {"row": "Family-related", "var": "family_loss_1_yr", **fit_reason(df, "family_loss_1_yr")},
    ]
    joint = fit_joint(df)

    body = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & $n_{\mathrm{treated}}$ & \% female & Heat x job loss & Stressor main \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Panel A: Separate regressions}} \\",
    ]
    for row in rows:
        c, s = cell(row["b"], row["se"], row["p"])
        m, ms = cell(row["stress_b"], row["stress_se"], row["stress_p"])
        body.append(rf"\quad {row['row']} & {row['n_treated']:,} & {row['female_pct']:.1f}\% & {c} & {m} \\")
        body.append(rf" &  &  & {s} & {ms} \\")
        body.append(r"\addlinespace[2pt]")
    body.append(r"\midrule")
    body.append(r"\multicolumn{5}{l}{\textit{Panel B: Joint subtype regression}} \\")
    for label, prefix in [("Heat x involuntary loss", "invol"), ("Heat x family-related loss", "family"), ("Contrast: involuntary - family", "diff")]:
        c, s = cell(joint[f"{prefix}_b"], joint[f"{prefix}_se"], joint[f"{prefix}_p"])
        body.append(rf"\quad {label} & --- & --- & {c} & --- \\")
        body.append(rf" &  &  & {s} &  \\")
    body.extend(
        [
            r"\midrule",
            r"Demographic controls & \multicolumn{4}{c}{Yes} \\",
            r"Kecamatan + Month + Year + Wave FE & \multicolumn{4}{c}{Yes} \\",
            r"\addlinespace[3pt]",
            rf"Observations & \multicolumn{{4}}{{c}}{{{int(rows[0]['n']):,}}} \\",
            r"Sample & \multicolumn{4}{c}{IFLS4-baseline panel} \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_outputs(TABLE, body, [*rows, {"row": "joint", **joint}])


if __name__ == "__main__":
    main()
