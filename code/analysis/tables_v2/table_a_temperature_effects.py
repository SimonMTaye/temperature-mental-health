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
    (r"\shortstack{7-day mean\\Wet Bulb}", "wetbulb_7d"),
    (r"\shortstack{Daily mean\\Temperature}", "tmean_c"),
    (r"\shortstack{Daily max\\Temperature}", "tmax_c"),
    (r"\shortstack{Daily mean\\Wet Bulb}", "wetbulb_c"),
]


def make_temp_table(
    outcome="cesd_z", filename="table_a_temperature_effects.tex"
) -> None:
    specs = []
    for label, term in OVERALL_HEAT_COLUMNS:
        spec = temperature_spec
        spec = update_formula_search_replace(spec, MAIN_TEMP_MEASURE, term)
        if outcome != "cesd_z":
            spec = update_formula_search_replace(spec, "cesd_z", outcome)
        specs.append(spec)

    table = make_regression_table(
        specs,
        rename=[{term: "heat"} for _, term in OVERALL_HEAT_COLUMNS],
        titles=[label for label, _ in OVERALL_HEAT_COLUMNS],
    )
    render_table_to_latex(table, TABLE_OUTPUT / filename)


def make_table():
    make_temp_table()
    make_temp_table(
        outcome="sleep_dur_h", filename="table_a2_temperature_effects_sleep.tex"
    )


if __name__ == "__main__":
    make_table()
