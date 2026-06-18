"""Table G: sleep mechanism using canonical sleep duration."""

from __future__ import annotations

from _lettered_common import CONTROLS, FE_IFLS5, cell, fit_model, load_analysis, multi_stats, write_outputs

TABLE = "table_g_mechanism_sleep"


def fit_sleep(data, formula_rhs: str, terms: dict[str, str], required: list[str]) -> dict[str, object]:
    formula = f"sleep_dur_h ~ {formula_rhs} + {CONTROLS} | {FE_IFLS5}"
    model = fit_model(data, formula, ["sleep_dur_h", "heat_c_dev", *required, "month", "year", "kecamatan_fe_code", "age", "female", "edu_yrs", "married", "widowed"])
    return multi_stats(model, terms)


def main() -> None:
    df = load_analysis()
    df = df[df["wave"] == "IFLS5"].copy()

    rows = [
        {"column": "Heat", **fit_sleep(df, "heat_c_dev", {"heat": "heat_c_dev"}, [])},
        {"column": "Job loss", **fit_sleep(df, "heat_c_dev * job_loss_1_yr", {"inter": "heat_c_dev:job_loss_1_yr", "heat": "heat_c_dev", "stress": "job_loss_1_yr"}, ["job_loss_1_yr"])},
        {"column": "Palm shock", **fit_sleep(df, "heat_c_dev * palm_farmer_hh_ifls4", {"inter": "heat_c_dev:palm_farmer_hh_ifls4", "heat": "heat_c_dev", "stress": "palm_farmer_hh_ifls4"}, ["palm_farmer_hh_ifls4"])},
        {"column": "Coal shock", **fit_sleep(df, "heat_c_dev * coal_worker_hh_ifls4", {"inter": "heat_c_dev:coal_worker_hh_ifls4", "heat": "heat_c_dev", "stress": "coal_worker_hh_ifls4"}, ["coal_worker_hh_ifls4"])},
        {"column": "Fuel cut", **fit_sleep(df, "heat_c_dev * post_subsidy * urban_vehicle_hh_ifls4", {"inter": "heat_c_dev:post_subsidy:urban_vehicle_hh_ifls4", "heat": "heat_c_dev", "stress": "urban_vehicle_hh_ifls4"}, ["post_subsidy", "urban_vehicle_hh_ifls4"])},
    ]

    inter_coef = [r"\textemdash", *[cell(row["inter_b"], row["inter_se"], row["inter_p"])[0] for row in rows[1:]]]
    inter_se = ["", *[cell(row["inter_b"], row["inter_se"], row["inter_p"])[1] for row in rows[1:]]]
    heat_coef = [cell(row["heat_b"], row["heat_se"], row["heat_p"])[0] for row in rows]
    heat_se = [cell(row["heat_b"], row["heat_se"], row["heat_p"])[1] for row in rows]

    body = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & (1) & (2) & (3) & (4) & (5) \\",
        r" & Heat & Job loss & Palm shock & Coal shock & Fuel cut \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Dependent variable: sleep duration in hours}} \\",
        "Heat x stressor & " + " & ".join(inter_coef) + r" \\",
        " & " + " & ".join(inter_se) + r" \\",
        r"\addlinespace[3pt]",
        "Heat & " + " & ".join(heat_coef) + r" \\",
        " & " + " & ".join(heat_se) + r" \\",
        r"\midrule",
        r"Demographic controls & Yes & Yes & Yes & Yes & Yes \\",
        r"Kecamatan FE & Yes & Yes & Yes & Yes & Yes \\",
        r"Month + Year FE & Yes & Yes & Yes & Yes & Yes \\",
        r"\addlinespace[3pt]",
        r"Sample & IFLS5 & IFLS5 & IFLS5 & IFLS5 & IFLS5 \\",
        "Observations & " + " & ".join(f"{int(row['n']):,}" for row in rows) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
