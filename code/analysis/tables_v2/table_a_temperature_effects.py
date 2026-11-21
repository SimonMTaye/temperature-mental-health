import numpy as np
from sharp_q_values import compute_q

from library.caching import run_regression_with_caching
from library.config import TABLE_OUTPUT
from library.specs import (
    temperature_spec,
    MAIN_TEMP_MEASURE,
    update_formula_search_replace,
)
from library.render import make_regression_table, render_table_to_latex

OVERALL_HEAT_COLUMNS = [
    (r"\shortstack{7-day mean\\Temperature}", "tmean_7d"),
    (r"\shortstack{>30 °C days\\in past week}", "hot30_7d"),
    (r"\shortstack{7-day mean\\Wet-bulb}", "wetbulb_7d"),
    (r"\shortstack{Daily mean\\Temperature}", "tmean_c"),
    (r"\shortstack{Daily max\\Temperature}", "tmax_c"),
    (r"\shortstack{Daily mean\\Wet-bulb}", "wetbulb_c"),
]


def sharpened_q_values(specs, terms) -> list[list[float]]:
    """Anderson (2008) sharpened FDR q-values, treating the table's heat coefficients as one family.

    Returns one list per model, aligned to that model's coefficients (NaN except the heat
    term), as maketables expects for custom_stats.
    """
    coef_tables = [run_regression_with_caching(spec).coef_table for spec in specs]
    p_values = [
        coef_table.loc[term, "p"]
        for coef_table, term in zip(coef_tables, terms, strict=True)
    ]
    q_values = compute_q(p_values)
    return [
        [q if name == term else np.nan for name in coef_table.index]
        for coef_table, term, q in zip(coef_tables, terms, q_values, strict=True)
    ]


def make_temp_table(
    outcome="cesd_z",
    filename="table_a_temperature_effects.tex",
    show_q_values=True,
) -> None:
    specs = []
    for label, term in OVERALL_HEAT_COLUMNS:
        spec = temperature_spec
        spec = update_formula_search_replace(spec, MAIN_TEMP_MEASURE, term)
        if outcome != "cesd_z":
            spec = update_formula_search_replace(spec, "cesd_z", outcome)
        specs.append(spec)

    terms = [term for _, term in OVERALL_HEAT_COLUMNS]
    q_value_kwargs = {}
    if show_q_values:
        # Braces stop LaTeX reading "\\ [q]" in the makecell as a line-spacing argument.
        q_value_kwargs = {
            "coef_fmt": "b:.3f* \n (se:.3f) \n {[q:.3f]}",
            "custom_stats": {"q": sharpened_q_values(specs, terms)},
        }

    table = make_regression_table(
        specs,
        rename=[{term: "heat"} for term in terms],
        titles=[label for label, _ in OVERALL_HEAT_COLUMNS],
        **q_value_kwargs,
    )
    render_table_to_latex(table, TABLE_OUTPUT / filename)


def make_table():
    make_temp_table()
    make_temp_table(
        outcome="sleep_dur_h",
        filename="table_a2_temperature_effects_sleep.tex",
        show_q_values=False,
    )


if __name__ == "__main__":
    make_table()
