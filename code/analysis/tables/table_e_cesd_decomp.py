"""Table E: CES-D factor decomposition."""

from __future__ import annotations

from _lettered_common import CONTROLS, FE_IFLS5, FE_POOLED, base_required, cell, fit_model, load_analysis, restrict_panel, term_stats, write_outputs

TABLE = "table_e_cesd_decomp"
DV_SPECS = [
    ("somatic_z", r"A. Somatic / activity-related"),
    ("depraffect_z", r"B. Depressed affect"),
    ("posaffect_z", r"C. Positive affect"),
]


def fit_row(data, formula: str, term: str, required: list[str]) -> dict[str, float]:
    return term_stats(fit_model(data, formula, required), term)


def main() -> None:
    df = restrict_panel(load_analysis())
    sub5 = df[df["wave"] == "IFLS5"].copy()
    rows = []
    for dv, _ in DV_SPECS:
        rows.extend(
            [
                {"dv": dv, "column": "Job loss", **fit_row(df, f"{dv} ~ heat_c_dev * job_loss_1_yr + {CONTROLS} | {FE_POOLED}", "heat_c_dev:job_loss_1_yr", [*base_required(dv), "job_loss_1_yr"])},
                {"dv": dv, "column": "Palm shock", **fit_row(df, f"{dv} ~ heat_c_dev * ifls5 * palm_farmer_hh_ifls4 + {CONTROLS} | {FE_POOLED}", "heat_c_dev:ifls5:palm_farmer_hh_ifls4", [*base_required(dv), "ifls5", "palm_farmer_hh_ifls4"])},
                {"dv": dv, "column": "Coal shock", **fit_row(df, f"{dv} ~ heat_c_dev * ifls5 * coal_worker_hh_ifls4 + {CONTROLS} | {FE_POOLED}", "heat_c_dev:ifls5:coal_worker_hh_ifls4", [*base_required(dv), "ifls5", "coal_worker_hh_ifls4"])},
                {"dv": dv, "column": "Fuel cut", **fit_row(sub5, f"{dv} ~ heat_c_dev * post_subsidy * urban_vehicle_hh_ifls4 + {CONTROLS} | {FE_IFLS5}", "heat_c_dev:post_subsidy:urban_vehicle_hh_ifls4", [dv, "heat_c_dev", "post_subsidy", "urban_vehicle_hh_ifls4", "month", "year", "kecamatan_code", "age", "female", "edu_yrs", "married", "widowed"])},
            ]
        )

    body = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & (1) & (2) & (3) & (4) \\",
        r" & Job loss & Palm shock & Coal shock & Fuel cut \\",
        r"\midrule",
    ]
    for dv, label in DV_SPECS:
        panel_rows = [row for row in rows if row["dv"] == dv]
        body.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{label}}}}} \\")
        body.append("Heat x stressor & " + " & ".join(cell(row["b"], row["se"], row["p"])[0] for row in panel_rows) + r" \\")
        body.append(" & " + " & ".join(cell(row["b"], row["se"], row["p"])[1] for row in panel_rows) + r" \\")
        body.append(r"\addlinespace[4pt]")
    body.extend(
        [
            r"\midrule",
            r"Demographic controls & Yes & Yes & Yes & Yes \\",
            r"Kecamatan FE & Yes & Yes & Yes & Yes \\",
            r"Month + Year FE & Yes & Yes & Yes & Yes \\",
            r"Wave FE & Yes & Yes & Yes & --- \\",
            r"\addlinespace[3pt]",
            r"Sample & Panel & Panel & Panel & IFLS5 panel \\",
            "Observations & " + " & ".join(f"{int(row['n']):,}" for row in rows[:4]) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
