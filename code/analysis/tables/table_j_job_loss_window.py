"""Table J: job-loss recall-window sensitivity."""

from __future__ import annotations

from _lettered_common import CONTROLS, FE_POOLED, cell, fit_model, load_analysis, restrict_ifls4_panel, term_stats, write_outputs

TABLE = "table_j_job_loss_window"
WINDOWS = [
    ("3 mo", "job_loss_90d"),
    ("6 mo", "job_loss_180d"),
    ("9 mo", "job_loss_270d"),
    ("12 mo", "job_loss_365d"),
    ("18 mo", "job_loss_540d"),
    ("24 mo", "job_loss_730d"),
    ("36 mo", "job_loss_1095d"),
    ("60 mo", "job_loss_1825d"),
]


def fit_window(df, variable: str) -> dict[str, object]:
    formula = f"cesd_z ~ heat_c_dev * {variable} + {CONTROLS} | {FE_POOLED}"
    required = ["cesd_z", "heat_c_dev", variable, "month", "year", "wave", "kecamatan_fe_code", "age", "female", "edu_yrs", "married", "widowed"]
    model = fit_model(df, formula, required)
    stats = term_stats(model, f"heat_c_dev:{variable}")
    frame = df.dropna(subset=required)
    return {
        **stats,
        "n_treated": int(frame[variable].sum()),
        "share": float(frame[variable].mean()),
        "stress_b": float(model.coef().get(variable, float("nan"))),
        "stress_se": float(model.se().get(variable, float("nan"))),
        "stress_p": float(model.pvalue().get(variable, float("nan"))),
    }


def main() -> None:
    df = restrict_ifls4_panel(load_analysis())
    rows = [{"label": label, "var": variable, **fit_window(df, variable)} for label, variable in WINDOWS]

    body = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Recall window & $n_{\mathrm{treated}}$ & Share & Heat x job loss & Stressor main \\",
        r"\midrule",
    ]
    for row in rows:
        c, s = cell(row["b"], row["se"], row["p"])
        m, ms = cell(row["stress_b"], row["stress_se"], row["stress_p"])
        body.append(rf"\quad Job loss within {row['label']} & {row['n_treated']:,} & {100 * row['share']:.1f}\% & {c} & {m} \\")
        body.append(rf" &  &  & {s} & {ms} \\")
        body.append(r"\addlinespace[2pt]")
    body.extend(
        [
            r"\midrule",
            r"Demographic controls & \multicolumn{4}{c}{Yes} \\",
            r"Kecamatan + Month + Year + Wave FE & \multicolumn{4}{c}{Yes} \\",
            r"\addlinespace[3pt]",
            rf"Observations & \multicolumn{{4}}{{c}}{{{int(rows[3]['n']):,}}} \\",
            r"Sample & \multicolumn{4}{c}{IFLS4-baseline panel} \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
