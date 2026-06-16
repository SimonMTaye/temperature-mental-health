"""Table C: cooling-degree-day robustness."""

from __future__ import annotations

from table_b_daynight import run_spec
from _lettered_common import CONTROLS, FE_IFLS5, FE_POOLED, base_required, cell, load_analysis, restrict_panel, write_outputs

TABLE = "table_c_cdd"
HEAT_SPECS = [
    ("cdd_tmax30", r"CDD Tmax $>$ 30$^\circ$C"),
    ("cdd_tmax32", r"CDD Tmax $>$ 32$^\circ$C"),
    ("cdd_tmin23", r"CDD Tmin $>$ 23$^\circ$C"),
    ("cdd_tmin24", r"CDD Tmin $>$ 24$^\circ$C"),
]


def main() -> None:
    df = restrict_panel(load_analysis())
    sub5 = df[df["wave"] == "IFLS5"].copy()
    rows = []
    for heat, _ in HEAT_SPECS:
        rows.extend(
            [
                run_spec(df, heat, "Job loss", df, f"cesd_z ~ {heat} * job_loss_1_yr + {CONTROLS} | {FE_POOLED}", f"{heat}:job_loss_1_yr", [*base_required("cesd_z", heat), "job_loss_1_yr"]),
                run_spec(df, heat, "Palm shock", df, f"cesd_z ~ {heat} * ifls5 * palm_farmer_hh_ifls4 + {CONTROLS} | {FE_POOLED}", f"{heat}:ifls5:palm_farmer_hh_ifls4", [*base_required("cesd_z", heat), "ifls5", "palm_farmer_hh_ifls4"]),
                run_spec(df, heat, "Coal shock", df, f"cesd_z ~ {heat} * ifls5 * coal_worker_hh_ifls4 + {CONTROLS} | {FE_POOLED}", f"{heat}:ifls5:coal_worker_hh_ifls4", [*base_required("cesd_z", heat), "ifls5", "coal_worker_hh_ifls4"]),
                run_spec(df, heat, "Fuel cut", sub5, f"cesd_z ~ {heat} * post_subsidy * urban_vehicle_hh_ifls4 + {CONTROLS} | {FE_IFLS5}", f"{heat}:post_subsidy:urban_vehicle_hh_ifls4", ["cesd_z", heat, "post_subsidy", "urban_vehicle_hh_ifls4", "month", "year", "kecamatan_fe_code", "age", "female", "edu_yrs", "married", "widowed"]),
            ]
        )

    body = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & (1) & (2) & (3) & (4) \\",
        r" & Job loss & Palm shock & Coal shock & Fuel cut \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Dependent variable: CES-D z-score}} \\",
    ]
    for heat, label in HEAT_SPECS:
        panel_rows = [row for row in rows if row["heat"] == heat]
        body.append(rf"\quad {label} x stressor & " + " & ".join(cell(row["b"], row["se"], row["p"])[0] for row in panel_rows) + r" \\")
        body.append(" & " + " & ".join(cell(row["b"], row["se"], row["p"])[1] for row in panel_rows) + r" \\")
        body.append(r"\addlinespace[3pt]")
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
