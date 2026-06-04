"""Table A: headline heat and baseline-stressor specifications."""

from __future__ import annotations

import pandas as pd

from _lettered_common import (
    BASELINE_GROUPS,
    CONTROLS,
    FE_IFLS5,
    FE_POOLED,
    base_required,
    cell,
    fit_model,
    load_analysis,
    restrict_panel,
    term_stats,
    write_outputs,
)

TABLE = "table_a_headline"


def fit_term(df: pd.DataFrame, formula: str, term: str, required: list[str]) -> dict[str, float]:
    # Headline identification follows the Mullins and White style comparison:
    # community FE plus month and year FE identify off interview-date weather
    # variation, conditional on survey timing within locations and seasons.
    model = fit_model(df, formula, required)
    return term_stats(model, term)


def main() -> None:
    df = restrict_panel(load_analysis())
    sub5 = df[df["wave"] == "IFLS5"].copy()

    specs = [
        (
            "Heat",
            df,
            f"cesd_z ~ heat_c_dev + {CONTROLS} | {FE_POOLED}",
            "heat_c_dev",
            base_required("cesd_z"),
        ),
        (
            "Heat x job loss",
            df,
            f"cesd_z ~ heat_c_dev * job_loss_1_yr + {CONTROLS} | {FE_POOLED}",
            "heat_c_dev:job_loss_1_yr",
            [*base_required("cesd_z"), "job_loss_1_yr"],
        ),
        (
            "Heat x IFLS5 x palm",
            df,
            f"cesd_z ~ heat_c_dev * ifls5 * palm_farmer_hh_ifls4 + {CONTROLS} | {FE_POOLED}",
            "heat_c_dev:ifls5:palm_farmer_hh_ifls4",
            [*base_required("cesd_z"), "ifls5", "palm_farmer_hh_ifls4"],
        ),
        (
            "Heat x IFLS5 x coal",
            df,
            f"cesd_z ~ heat_c_dev * ifls5 * coal_worker_hh_ifls4 + {CONTROLS} | {FE_POOLED}",
            "heat_c_dev:ifls5:coal_worker_hh_ifls4",
            [*base_required("cesd_z"), "ifls5", "coal_worker_hh_ifls4"],
        ),
        (
            "Heat x subsidy cut x urban vehicle",
            sub5,
            f"cesd_z ~ heat_c_dev * post_subsidy * urban_vehicle_hh_ifls4 + {CONTROLS} | {FE_IFLS5}",
            "heat_c_dev:post_subsidy:urban_vehicle_hh_ifls4",
            [
                "cesd_z",
                "heat_c_dev",
                "post_subsidy",
                "urban_vehicle_hh_ifls4",
                "month",
                "year",
                "kecamatan_fe_code",
                *BASELINE_GROUPS,
                *base_required("cesd_z")[6:],
            ],
        ),
    ]

    results = []
    for label, data, formula, term, required in specs:
        results.append({"label": label, "term": term, **fit_term(data, formula, term, required)})

    coef_cells = [cell(row["b"], row["se"], row["p"])[0] for row in results]
    se_cells = [cell(row["b"], row["se"], row["p"])[1] for row in results]

    body = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & (1) & (2) & (3) & (4) & (5) \\",
        r" & Heat & Job loss & Palm shock & Coal shock & Fuel cut \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Dependent variable: CES-D z-score}} \\",
        r"\addlinespace[2pt]",
        "Key heat coefficient & " + " & ".join(coef_cells) + r" \\",
        " & " + " & ".join(se_cells) + r" \\",
        r"\midrule",
        r"Demographic controls & Yes & Yes & Yes & Yes & Yes \\",
        r"Kecamatan FE & Yes & Yes & Yes & Yes & Yes \\",
        r"Month + Year FE & Yes & Yes & Yes & Yes & Yes \\",
        r"Wave FE & Yes & Yes & Yes & Yes & --- \\",
        r"\addlinespace[3pt]",
        r"Sample & Panel & Panel & Panel & Panel & IFLS5 panel \\",
        "Observations & " + " & ".join(f"{int(row['n']):,}" for row in results) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_outputs(TABLE, body, results)


if __name__ == "__main__":
    main()
