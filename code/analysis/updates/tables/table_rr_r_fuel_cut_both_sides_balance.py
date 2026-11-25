"""Balance test: kecamatans interviewed on both sides of the fuel cut vs. after only.

Run from the repo root:

    uv run python code/analysis/updates/tables/table_rr_r_fuel_cut_both_sides_balance.py

Writes output/tables/table_rr_r_fuel_cut_both_sides_balance.tex.

Companion to table_rr_o_fuel_cut_both_sides_table.py. With kecamatan fixed
effects, the fuel-shock term is identified only from kecamatans with IFLS5
interviews both before and after the Nov 18, 2014 subsidy cut. This table asks
whether respondents in those kecamatans look like respondents in kecamatans
interviewed entirely after the cut. Kecamatans interviewed entirely before the
cut (about 2% of IFLS5 respondents) are left out.

Rows follow Table C (table_c_shock_balances.py): demographics, cash-transfer
receipt and the winsorized economic outcomes, plus the urban-vehicle share and
7-day mean temperature. Columns (1)-(2) are group means (SD). Column (3) is the
difference in means, both sides minus after only. With about 30,000
respondents, small imbalances are statistically significant, so the table
reports magnitudes rather than tests. There are no fixed effects: both-sides status is fixed within a kecamatan, so kecamatan fixed
effects would absorb it. These are descriptive differences, not tests of the
identifying assumption, which conditions on kecamatan and month-by-year fixed
effects.
"""

import pandas as pd
from analysis.tables_v2.table_c_shock_balances import balance_variables
from analysis.tables_v2.table_i_economics import ECONOMIC_OUTCOMES, winsorized
from library.config import TABLE_OUTPUT
from library.dictionary import VARIABLE_LABELS
from library.specs import MAIN_TEMP_MEASURE, wave5_df
from library.table_builder import coefficient_rows, make_row

GROUP = "both_sides_kecamatan"
URBAN_VEHICLE = "urban_vehicle_hh_ifls4"
ROW_VARIABLES = [*balance_variables, URBAN_VEHICLE, MAIN_TEMP_MEASURE]
ROW_LABELS = {
    **VARIABLE_LABELS,
    URBAN_VEHICLE: "Urban vehicle owner",
    "expenditure_transport_fuel_total_mo_real_usd": "Transport Spending (2010 USD)",
    MAIN_TEMP_MEASURE: "7-day mean temperature (°C)",
}

TABLE_TEMPLATE = r"""
\begin{tabular}{@{}lccc}
\toprule
 & \multicolumn{2}{c}{Kecamatans Surveyed -- IFLS5} & \\
\cmidrule(lr){2-3}
 & \shortstack{Before and After\\Subsidy Cut} & \shortstack{After Subsidy\\Cut Only} & Difference \\
 & (1) & (2) & (3) \\
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
    # The table reports no tests, so a NaN p-value suppresses stars.
    return (series.mean(), series.std(), float("nan"))


def balance_rows(sample: pd.DataFrame) -> list[str]:
    both_sides = sample.loc[sample[GROUP].eq(1)]
    after_only = sample.loc[sample[GROUP].eq(0)]

    rows = []
    for variable in ROW_VARIABLES:
        mean_row, sd_row = coefficient_rows(
            ROW_LABELS[variable],
            [mean_sd(both_sides[variable]), mean_sd(after_only[variable])],
        )
        difference = both_sides[variable].mean() - after_only[variable].mean()
        # Column (3) has only the difference in means; its SD row is left blank.
        rows.extend(
            [
                mean_row.removesuffix(r" \\") + f" & {difference:.3f} \\\\",
                sd_row.removesuffix(r" \\") + r" & \\",
            ]
        )
    return rows


def count_row(label: str, sample: pd.DataFrame, count) -> str:
    return make_row(
        label,
        [
            count(sample.loc[sample[GROUP].eq(1)]),
            count(sample.loc[sample[GROUP].eq(0)]),
            count(sample),
        ],
    )


def build_table(sample: pd.DataFrame) -> str:
    return (
        TABLE_TEMPLATE.replace("{coefficient_rows}", "\n".join(balance_rows(sample)))
        .replace("{observations_row}", count_row("Observations", sample, len))
        .replace(
            "{kecamatans_row}",
            count_row("Kecamatans", sample, lambda df: df["gadm_fullcode"].nunique()),
        )
    )


def main() -> None:
    sample = both_sides_or_after_only(wave5_df)

    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    (TABLE_OUTPUT / "table_rr_r_fuel_cut_both_sides_balance.tex").write_text(
        build_table(sample), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
