"""Balance test: kecamatans interviewed on both sides of the fuel cut vs. after only.

Run from the repo root:

    uv run python code/analysis/updates/tables/table_rr_r_fuel_cut_both_sides_balance.py

Writes output/tables/table_rr_r_fuel_cut_both_sides_balance.tex (cell values in
output/stats/table_rr_r/).

Companion to table_rr_o_fuel_cut_both_sides_table.py. With kecamatan fixed
effects, the fuel-shock term is identified only from kecamatans with IFLS5
interviews both before and after the Nov 18, 2014 subsidy cut. This table asks
whether respondents in those kecamatans look like respondents in kecamatans
interviewed entirely after the cut. Kecamatans interviewed entirely before the
cut (about 2% of IFLS5 respondents) are left out.

Rows follow Table C (table_c_shock_balances.py): demographics, cash-transfer
receipt and the winsorized economic outcomes, plus the urban-vehicle share and
7-day mean temperature. Columns (1)-(2) are group means (SD). Columns (3)-(4)
regress each variable on a both-sides indicator, for all IFLS5 respondents and
for IFLS4 urban-vehicle households only (the fuel-shock group). There are no
fixed effects: both-sides status is fixed within a kecamatan, so kecamatan
fixed effects would absorb it. Standard errors are clustered by kabupaten.
These are descriptive differences, not tests of the identifying assumption,
which conditions on kecamatan and month-by-year fixed effects.
"""

import pandas as pd
from analysis.tables_v2.table_c_shock_balances import balance_variables
from analysis.tables_v2.table_i_economics import ECONOMIC_OUTCOMES, winsorized
from library.caching import run_regression_with_caching
from library.config import TABLE_OUTPUT
from library.dictionary import VARIABLE_LABELS
from library.render import RegressionSpec
from library.specs import MAIN_TEMP_MEASURE, wave5_df
from library.table_builder import coefficient_rows, make_row
from library.table_stats import (
    StatsGrid,
    bottom_row,
    regression_cell,
    write_table_stats,
)

GROUP = "both_sides_kecamatan"
URBAN_VEHICLE = "urban_vehicle_hh_ifls4"
ROW_VARIABLES = [*balance_variables, URBAN_VEHICLE, MAIN_TEMP_MEASURE]
ROW_LABELS = {
    **VARIABLE_LABELS,
    URBAN_VEHICLE: "Urban vehicle owner",
    MAIN_TEMP_MEASURE: "7-day mean temperature (°C)",
}

TABLE_TEMPLATE = r"""
\begin{tabular}{@{}lcccc}
\toprule
 & \multicolumn{2}{c}{Mean (SD)} & \multicolumn{2}{c}{Difference: Both Sides $-$ After Only} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
 & Both Sides & After Only & All Respondents & \shortstack{Urban Vehicle\\Owners} \\
 & (1) & (2) & (3) & (4) \\
\midrule\addlinespace[2.5pt]
{coefficient_rows}
\midrule\addlinespace[2.5pt]
{observations_row}
{kecamatans_row}
\bottomrule
\end{tabular}
"""


def both_sides_or_after_only(wave5: pd.DataFrame) -> pd.DataFrame:
    """Keep kecamatans with interviews on both sides of the cut or after it only."""
    share_after = (
        wave5.groupby("gadm_fullcode")["post_subsidy"].transform("mean").astype(float)
    )
    sample = wave5.loc[share_after > 0].copy()
    sample[GROUP] = share_after.loc[sample.index].lt(1).astype(int)
    for variable in ECONOMIC_OUTCOMES:
        sample[variable] = winsorized(sample[variable])
    return sample


def mean_sd(series: pd.Series) -> tuple[float, float, float]:
    # p-value 0 suppresses stars, as in Table C.
    return (series.mean(), series.std(), 0)


def difference(variable: str, df: pd.DataFrame) -> tuple[float, float, float]:
    spec = RegressionSpec(
        title=f"{variable} balance on {GROUP}",
        formula=f"{variable} ~ {GROUP}",
        df=df,
        tags=frozenset(),
        show_terms=frozenset([GROUP]),
    )
    result = run_regression_with_caching(spec)
    return tuple(float(result.coef_table.loc[GROUP, stat]) for stat in ["b", "se", "p"])


def balance_stats(sample: pd.DataFrame) -> dict[str, list[tuple[float, float, float]]]:
    """Per row variable: both-sides mean (SD), after-only mean (SD), differences."""
    both_sides = sample.loc[sample[GROUP].eq(1)]
    after_only = sample.loc[sample[GROUP].eq(0)]
    urban_vehicle_owners = sample.loc[sample[URBAN_VEHICLE].eq(1)]

    stats_by_variable = {}
    for variable in ROW_VARIABLES:
        stats = [
            mean_sd(both_sides[variable]),
            mean_sd(after_only[variable]),
            difference(variable, sample),
        ]
        # Urban-vehicle status is constant within urban-vehicle owners, so
        # column (4) is left blank for that row.
        if variable != URBAN_VEHICLE:
            stats.append(difference(variable, urban_vehicle_owners))
        stats_by_variable[variable] = stats
    return stats_by_variable


def balance_rows(stats_by_variable: dict[str, list]) -> list[str]:
    rows = []
    for variable, stats in stats_by_variable.items():
        coefficient_row, standard_error_row = coefficient_rows(ROW_LABELS[variable], stats)
        if len(stats) == 3:
            coefficient_row = coefficient_row.replace(r" \\", r" & \\")
            standard_error_row = standard_error_row.replace(r" \\", r" & \\")
        rows.extend([coefficient_row, standard_error_row])
    return rows


def group_counts(sample: pd.DataFrame, count) -> list[int]:
    """Counts for columns (1)-(4): both sides, after only, all, urban-vehicle owners."""
    return [
        int(count(sample.loc[sample[GROUP].eq(1)])),
        int(count(sample.loc[sample[GROUP].eq(0)])),
        int(count(sample)),
        int(count(sample.loc[sample[URBAN_VEHICLE].eq(1)])),
    ]


def count_kecamatans(df: pd.DataFrame) -> int:
    return df["gadm_fullcode"].nunique()


def build_table(stats_by_variable: dict[str, list], sample: pd.DataFrame) -> str:
    return (
        TABLE_TEMPLATE.replace(
            "{coefficient_rows}", "\n".join(balance_rows(stats_by_variable))
        )
        .replace(
            "{observations_row}", make_row("Observations", group_counts(sample, len))
        )
        .replace(
            "{kecamatans_row}",
            make_row("Kecamatans", group_counts(sample, count_kecamatans)),
        )
    )


def table_stats(stats_by_variable: dict[str, list], sample: pd.DataFrame) -> StatsGrid:
    """Columns (1)-(2) are mean (SD); (3)-(4) are differences; then the count rows."""
    grid: StatsGrid = [
        [
            {"mean": both_mean, "sd": both_sd},
            {"mean": after_mean, "sd": after_sd},
            *(regression_cell(*difference_stats) for difference_stats in differences),
        ]
        for (both_mean, both_sd, _), (after_mean, after_sd, _), *differences in (
            stats_by_variable.values()
        )
    ]
    grid.append(bottom_row(group_counts(sample, len)))
    grid.append(bottom_row(group_counts(sample, count_kecamatans)))
    return grid


def main() -> None:
    sample = both_sides_or_after_only(wave5_df)
    stats_by_variable = balance_stats(sample)

    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    (TABLE_OUTPUT / "table_rr_r_fuel_cut_both_sides_balance.tex").write_text(
        build_table(stats_by_variable, sample), encoding="utf-8"
    )
    write_table_stats("rr_r", table_stats(stats_by_variable, sample))


if __name__ == "__main__":
    main()
