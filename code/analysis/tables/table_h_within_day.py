"""Table H: within-day hourly temperature specification."""

from __future__ import annotations

from _lettered_common import CONTROLS, FE_WITHIN_DAY, cell, fit_model, load_analysis, restrict_panel, term_stats, write_outputs

TABLE = "table_h_within_day"


def fit_term(data, formula: str, term: str, required: list[str]) -> dict[str, object]:
    model = fit_model(data, formula, required, singleton_columns=("kecamatan_code", "day_id"))
    return term_stats(model, term)


def main() -> None:
    df = restrict_panel(load_analysis())
    if df["heat_hr_dev"].notna().sum() == 0 or df["tmean_c_hour"].notna().sum() == 0:
        raise ValueError("table_h cannot run: canonical hourly columns tmean_c_hour and heat_hr_dev are entirely missing")
    sub5 = df[df["wave"] == "IFLS5"].copy()
    required = ["cesd_z", "heat_hr_dev", "day_id", "kecamatan_code", "age", "female", "edu_yrs", "married", "widowed"]
    specs = [
        ("Heat", df, f"cesd_z ~ heat_hr_dev + {CONTROLS} | {FE_WITHIN_DAY}", "heat_hr_dev", required),
        ("Job loss", df, f"cesd_z ~ heat_hr_dev * job_loss_1_yr + {CONTROLS} | {FE_WITHIN_DAY}", "heat_hr_dev:job_loss_1_yr", [*required, "job_loss_1_yr"]),
        ("Palm shock", df, f"cesd_z ~ heat_hr_dev * ifls5 * palm_farmer_hh_ifls4 + {CONTROLS} | {FE_WITHIN_DAY}", "heat_hr_dev:ifls5:palm_farmer_hh_ifls4", [*required, "ifls5", "palm_farmer_hh_ifls4"]),
        ("Coal shock", df, f"cesd_z ~ heat_hr_dev * ifls5 * coal_worker_hh_ifls4 + {CONTROLS} | {FE_WITHIN_DAY}", "heat_hr_dev:ifls5:coal_worker_hh_ifls4", [*required, "ifls5", "coal_worker_hh_ifls4"]),
        ("Fuel cut", sub5, f"cesd_z ~ heat_hr_dev * post_subsidy * urban_vehicle_hh_ifls4 + {CONTROLS} | {FE_WITHIN_DAY}", "heat_hr_dev:post_subsidy:urban_vehicle_hh_ifls4", [*required, "post_subsidy", "urban_vehicle_hh_ifls4"]),
    ]
    rows = [{"column": label, "term": term, **fit_term(data, formula, term, req)} for label, data, formula, term, req in specs]
    coef = [cell(row["b"], row["se"], row["p"])[0] for row in rows]
    se = [cell(row["b"], row["se"], row["p"])[1] for row in rows]
    body = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & (1) & (2) & (3) & (4) & (5) \\",
        r" & Heat & Job loss & Palm shock & Coal shock & Fuel cut \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Dependent variable: CES-D z-score}} \\",
        r"\multicolumn{6}{l}{\textit{Heat measured at the survey hour}} \\",
        "Key hourly heat coefficient & " + " & ".join(coef) + r" \\",
        " & " + " & ".join(se) + r" \\",
        r"\midrule",
        r"Demographic controls & Yes & Yes & Yes & Yes & Yes \\",
        r"Kecamatan FE & Yes & Yes & Yes & Yes & Yes \\",
        r"Calendar-day FE & Yes & Yes & Yes & Yes & Yes \\",
        r"\addlinespace[3pt]",
        r"Sample & Panel & Panel & Panel & Panel & IFLS5 panel \\",
        "Observations & " + " & ".join(f"{int(row['n']):,}" for row in rows) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
