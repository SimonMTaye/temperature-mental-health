"""Re-run Table A with a quadratic in heat to test for non-linear temperature effects.

Run from the repo root:

    uv run python code/analysis/updates/tables/table_rr_s_quadratic_temperature.py

Writes:
    output/tables/table_rr_s_quadratic_temperature.tex        (CES-D z-score)
    output/tables/table_rr_s2_quadratic_temperature_sleep.tex (sleep hours)
    (cell values in output/stats/table_rr_s/ and output/stats/table_rr_s2/)
and prints the linear and squared coefficients for a quick read.

Columns are the six Table A heat measures. Each column regresses the outcome on
heat and heat squared, where heat is demeaned at its analysis-sample mean
before squaring. So:
    Heat     = slope at the sample-mean heat level (comparable to Table A's
               linear coefficient);
    Heat^2   = curvature. A positive value means heat hurts more at the hot
               end; its t-test is the test for a quadratic effect.

Identification is the same as Table A (Mullins & White 2019-style): kecamatan
and month-by-year fixed effects, so heat effects come from daily/weekly weather
deviations around local and calendar norms, assumed as good as random
conditional on those fixed effects and the demographic controls.

The CES-D table reports Anderson (2008) sharpened FDR q-values in brackets,
treating all twelve Heat and Heat^2 coefficients as a single family.
"""

import numpy as np
import pandas as pd
from analysis.tables_v2.table_a_temperature_effects import OVERALL_HEAT_COLUMNS
from sharp_q_values import compute_q

from library.caching import run_regression_with_caching
from library.config import TABLE_OUTPUT
from library.dictionary import VARIABLE_LABELS
from library.render import (
    RegressionSpec,
    make_regression_table,
    render_table_to_latex,
)
from library.specs import CONTROLS, FE_NO_WAVE, analysis_df
from library.table_stats import write_table_stats

LABELS = {
    **VARIABLE_LABELS,
    "heat": "Heat",
    "heat_sq": "Heat$^2$",
}


def add_demeaned_heat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add {term}_dm and {term}_dm_sq for every Table A heat measure."""
    new_columns = {}
    for _, term in OVERALL_HEAT_COLUMNS:
        demeaned = df[term] - df[term].mean()
        new_columns[f"{term}_dm"] = demeaned
        new_columns[f"{term}_dm_sq"] = demeaned**2
    return df.assign(**new_columns)


quadratic_df = add_demeaned_heat_columns(analysis_df)


def quadratic_spec(outcome: str, term: str) -> RegressionSpec:
    # Same controls and FE as Table A (kecamatan + month*year); identification
    # comes from weather deviations around local and calendar norms.
    return RegressionSpec(
        title="",
        formula=f"{outcome} ~ {term}_dm + {term}_dm_sq + {CONTROLS} | {FE_NO_WAVE}",
        df=quadratic_df,
        tags=frozenset(["temperature-effect", "quadratic"]),
        show_terms=frozenset([f"{term}_dm", f"{term}_dm_sq"]),
    )


def sharpened_q_values(specs, terms) -> list[list[float]]:
    """Sharpened FDR q-values, pooling all twelve Heat and Heat^2 terms into one family.

    Returns one list per model aligned to its coefficients (NaN for controls),
    as maketables expects for custom_stats.
    """
    coef_tables = [run_regression_with_caching(spec).coef_table for spec in specs]
    family = [
        (coef_table, f"{term}{suffix}")
        for coef_table, term in zip(coef_tables, terms, strict=True)
        for suffix in ["_dm", "_dm_sq"]
    ]
    p_values = [coef_table.loc[name, "p"] for coef_table, name in family]
    q_by_term = {
        name: q for (_, name), q in zip(family, compute_q(p_values), strict=True)
    }
    return [
        [q_by_term.get(name, np.nan) for name in coef_table.index]
        for coef_table in coef_tables
    ]


def make_quadratic_table(
    outcome: str, filename: str, stats_name: str, show_q_values: bool
) -> None:
    terms = [term for _, term in OVERALL_HEAT_COLUMNS]
    specs = [quadratic_spec(outcome, term) for term in terms]

    q_value_kwargs = {}
    if show_q_values:
        # Braces stop LaTeX reading "\\ [q]" in the makecell as a line-spacing argument.
        q_value_kwargs = {
            "coef_fmt": "b:.3f* \n (se:.3f) \n {[q:.3f]}",
            "custom_stats": {"q": sharpened_q_values(specs, terms)},
        }

    table = make_regression_table(
        specs,
        rename=[{f"{term}_dm_sq": "heat_sq", f"{term}_dm": "heat"} for term in terms],
        titles=[label for label, _ in OVERALL_HEAT_COLUMNS],
        keep=["heat", "heat_sq"],
        order=["heat", "heat_sq"],
        exact_match=True,
        labels=LABELS,
        **q_value_kwargs,
    )
    render_table_to_latex(table, TABLE_OUTPUT / filename)
    write_table_stats(stats_name, table.coefficient_stats)


def print_summary(outcome: str) -> None:
    rows = []
    for label, term in OVERALL_HEAT_COLUMNS:
        coef_table = run_regression_with_caching(
            quadratic_spec(outcome, term)
        ).coef_table
        for suffix in ["_dm", "_dm_sq"]:
            b, se, p = coef_table.loc[f"{term}{suffix}", ["b", "se", "p"]]
            rows.append({"heat": term, "term": suffix, "b": b, "se": se, "p": p})
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(f"Quadratic heat effects on {outcome}:")
        print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    make_quadratic_table(
        outcome="cesd_z",
        filename="table_rr_s_quadratic_temperature.tex",
        stats_name="rr_s",
        show_q_values=True,
    )
    make_quadratic_table(
        outcome="sleep_dur_h",
        filename="table_rr_s2_quadratic_temperature_sleep.tex",
        stats_name="rr_s2",
        show_q_values=False,
    )
    print_summary("cesd_z")
    print_summary("sleep_dur_h")


if __name__ == "__main__":
    main()
