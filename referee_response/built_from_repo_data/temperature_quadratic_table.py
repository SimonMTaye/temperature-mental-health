"""Average effect of heat on CES-D with a quadratic temperature term.

Run from the repo root:

    uv run python code/analysis/updates/temperature_quadratic_table.py

Responds to Reviewer 2, minor comment 2. Writes
output/tables/table_temperature_quadratic.tex.

Columns (1) and (3) are the paper's linear main-effect regressions
(temperature_spec) for mean and wet-bulb temperature. Columns (2) and (4) add
the square of temperature. Temperature is centered at the sample median before
squaring, so the linear coefficient is the slope at the median. The last rows
give p-values for the squared term and for all temperature terms jointly.
Controls, fixed effects and district clustering are the same as in the paper.
"""

from dataclasses import replace

import numpy as np
from library.caching import run_regression_with_caching
from library.config import TABLE_OUTPUT
from library.specs import MAIN_TEMP_MEASURE, temperature_spec
from library.table_builder import coefficient_rows, make_row
from scipy import stats

WETBULB_MEASURE = "wetbulb_7d"
HEAT = "heat"
HEAT_SQUARED = "heat_squared"

TABLE_TEMPLATE = r"""
\begin{tabular}{@{}lcccc}
\toprule
 & \multicolumn{4}{c}{CES-D z-score} \\
\cmidrule(lr){2-5}
 & \multicolumn{2}{c}{7-day mean Temperature} & \multicolumn{2}{c}{7-day mean Wet-bulb} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
 & Linear & Quadratic & Linear & Quadratic \\
 & (1) & (2) & (3) & (4) \\
\midrule\addlinespace[2.5pt]
{coefficient_rows}
\midrule\addlinespace[2.5pt]
{p_value_rows}
\midrule\addlinespace[2.5pt]
Kecamatan & x & x & x & x \\
Month-Year & x & x & x & x \\
\midrule\addlinespace[2.5pt]
{observations_row}
\bottomrule
\end{tabular}
"""


def heat_specs(measure: str) -> list:
    """Linear and quadratic specs, with the measure renamed to heat and centered."""
    df = temperature_spec.df.copy()
    used = df.dropna(subset=["cesd_z", measure])
    df[HEAT] = df[measure] - used[measure].median()
    df[HEAT_SQUARED] = df[HEAT] ** 2
    linear = replace(
        temperature_spec,
        formula=temperature_spec.formula.replace(MAIN_TEMP_MEASURE, HEAT),
        df=df,
    )
    quadratic = replace(
        linear, formula=linear.formula.replace(HEAT, f"{HEAT} + {HEAT_SQUARED}", 1)
    )
    return [linear, quadratic]


def wald_p_value(model, terms: list[str]) -> float:
    """Joint test that the listed coefficients are zero, district-clustered."""
    terms = [term for term in terms if term in model.coef_table.index]
    b = model.coef_table.loc[terms, "b"].to_numpy()
    v = model.vcov.loc[terms, terms].to_numpy()
    f_statistic = float(b @ np.linalg.solve(v, b)) / len(terms)
    return float(stats.f.sf(f_statistic, len(terms), model.stats["n_clusters"] - 1))


def stats_or_blank(model, term: str) -> tuple:
    if term not in model.coef_table.index:
        return (np.nan, np.nan, np.nan)
    return tuple(float(model.coef_table.loc[term, key]) for key in ["b", "se", "p"])


def build_table() -> str:
    models = [
        run_regression_with_caching(spec)
        for measure in [MAIN_TEMP_MEASURE, WETBULB_MEASURE]
        for spec in heat_specs(measure)
    ]
    rows = []
    for label, term in [("Heat", HEAT), ("Heat Squared", HEAT_SQUARED)]:
        coefs, ses = coefficient_rows(
            label, [stats_or_blank(model, term) for model in models]
        )
        # Linear columns have no squared term: leave those cells empty.
        rows += [coefs.replace("nan", ""), ses.replace("(nan)", "")]
    squared_p_values = [
        f"{wald_p_value(model, [HEAT_SQUARED]):.3f}"
        if HEAT_SQUARED in model.coef_table.index
        else ""
        for model in models
    ]
    joint_p_values = [
        f"{wald_p_value(model, [HEAT, HEAT_SQUARED]):.3f}" for model in models
    ]
    return (
        TABLE_TEMPLATE.replace("{coefficient_rows}", "\n".join(rows))
        .replace(
            "{p_value_rows}",
            make_row("p-value: Heat Squared = 0", squared_p_values)
            + "\n"
            + make_row("p-value: All Heat Terms = 0", joint_p_values),
        )
        .replace(
            "{observations_row}",
            make_row("Observations", [int(model.stats["N"]) for model in models]),
        )
    )


def main() -> None:
    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    (TABLE_OUTPUT / "table_temperature_quadratic.tex").write_text(
        build_table(), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
