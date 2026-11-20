from library.config import TABLE_OUTPUT
from library.specs import (
    palm_shock,
    build_palm_df,
    update_formula_search_replace,
    MAIN_TEMP_MEASURE,
)
from library.render import make_shock_regression_table_trimmed, render_table_to_latex
from dataclasses import replace

GROUPS = [
    ("Main Definition", "palm_farmer_hh_ifls4"),
    (r"Both IFLS4\&5", "palm_farmer_both_waves"),
    ("IFLS5", "palm_farmer_hh_ifls5"),
    # (r"\shortstack{Farmer non-palm\\region}", "farmer_non_palm_ifls4"),
]


def make_specs(palm_df):
    """Return the palm-shock spec once per alternative palm-farmer definition."""
    return [
        replace(
            update_formula_search_replace(palm_shock, "palm_farmer_hh_ifls4", term),
            df=palm_df,
        )
        for _, term in GROUPS
    ]


def make_table() -> None:
    specs = make_specs(build_palm_df())
    table = make_shock_regression_table_trimmed(
        specs,
        group=[term for _, term in GROUPS],
        post="ifls5",
        temperature=MAIN_TEMP_MEASURE,
        titles=[label for label, _ in GROUPS],
    )
    render_table_to_latex(
        table, TABLE_OUTPUT / "table_d_palm_farmer_alt_definition.tex"
    )


if __name__ == "__main__":
    make_table()
