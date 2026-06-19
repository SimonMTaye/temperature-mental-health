from library.config import TABLE_OUTPUT
from library.specs import (
    palm_shock,
    analysis_df,
    update_formula_search_replace,
    MAIN_TEMP_MEASURE,
)
from library.render import make_shock_regression_table_trimmed, render_table_to_latex
from dataclasses import replace


def make_table() -> None:
    wave4 = (
        analysis_df.copy()
        .query("wave == 'IFLS4'")
        .filter(items=["pidlink", "job_self_employed"])
        .rename(columns={"job_self_employed": "job_self_employed_ifls4"})
    )
    palm_df = (
        analysis_df.copy()
        .merge(wave4, on="pidlink", how="left", validate="m:1")
        .assign(
            palm_farmer_self_employed=lambda df: (
                df["palm_farmer_hh_ifls4"] & df["job_self_employed_ifls4"]
            ).astype("Int32"),
            palm_farmer_wage=lambda df: (
                df["palm_farmer_hh_ifls4"] & ~df["job_self_employed_ifls4"]
            ).astype("Int32"),
        )
    )

    groups = [
        ("Palm Farmers", "palm_farmer_hh_ifls4"),
        (r"\shortstack{Palm Farmer\\Self-Employed}", "palm_farmer_self_employed"),
        (r"\shortstack{Palm Farmer\\Wage}", "palm_farmer_wage"),
    ]
    specs = [
        replace(
            update_formula_search_replace(
                palm_shock,
                "palm_farmer_hh_ifls4",
                term,
            ),
            df=palm_df,
        )
        for _, term in groups
    ]
    treated = [palm_df[group_var].eq(1).sum() for _, group_var in groups]

    table = make_shock_regression_table_trimmed(
        specs,
        group=[term for _, term in groups],
        post="ifls5",
        temperature=MAIN_TEMP_MEASURE,
        titles=[label for label, _ in groups],
        custom_model_stats={"Palm Farmer Count": treated},
    )
    render_table_to_latex(table, TABLE_OUTPUT / "table_j_palm_farmer_selfemp.tex")


if __name__ == "__main__":
    make_table()
