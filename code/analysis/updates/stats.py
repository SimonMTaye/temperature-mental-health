"""Derived statistics quoted in the paper, computed from the table stats files.

Run from the repo root, after the tables:

    uv run python -m analysis.tables_v2.main
    uv run python code/analysis/updates/stats.py

Assumptions:
- Every table in analysis/tables_v2 has been run, so output/stats/table_<x>/
  holds one file per displayed cell (see library/table_stats.py for the
  table_<x>_<stat>_c<column>_r<row>.tex naming).
- Columns and rows below refer to the current table layouts, e.g. Table B
  column 3 is the continuous palm price drop and column 6 the cash-transfer
  recipients; Table B row 1 is the 7-day temperature differential, row 3 the
  wet-bulb differential and row 6 the stressed-group proportion. If a table's
  layout changes, the matching functions here must change too.
- Two inputs are not in any table and are set as constants below.

Each function listed in DERIVED_STATS computes one statistic. Its name is
the output file name:
output/stats/derived/<function name>.tex. Names follow
table_<letter>_<operation>_<subject>, where <operation> is one of ci95_lower,
ci95_upper, count, min, max, percent, percent_of_mean, percent_difference,
percent_gap, abs, minutes, population_effect or warm_week_effect.
"""

from library.config import STATS_OUTPUT
from library.table_stats import format_stat

# Gap in 7-day mean temperature between the 90th and 10th percentile week
# (degrees C), from analysis/updates/one-off/residual_temperature_magnitudes.py.
TEMPERATURE_P90_P10_GAP = 1.13
# Size of the November 2014 subsidized fuel price increase (percent); policy fact.
FUEL_PRICE_INCREASE_PERCENT = 30

DERIVED_OUTPUT = STATS_OUTPUT / "derived"

TABLE_A_HEAT_MEASURE_COLUMNS = [1, 2, 3, 4, 5, 6]
TABLE_A_7DAY_TEMPERATURE_COLUMN = 1
TABLE_B_COLUMNS = [1, 2, 3, 4, 5, 6, 7]
TABLE_B_CASH_TRANSFER_COLUMN = 6
TABLE_B_BINARY_SHOCK_COLUMNS = [1, 2, 4, 5, 7]
TABLE_B_TEMPERATURE_DIFFERENTIAL_ROW = 1
TABLE_B_WETBULB_DIFFERENTIAL_ROW = 3
TABLE_B_STRESSED_PROPORTION_ROW = 6
TABLE_B_PALM_COLUMN = 1


# ---------------------------------------------------------------------------
# Helpers (not written to output/stats/derived)
# ---------------------------------------------------------------------------


def stat(table: str, name: str, column: int, row: int) -> float:
    path = (
        STATS_OUTPUT / f"table_{table}" / f"table_{table}_{name}_c{column}_r{row}.tex"
    )
    return float(path.read_text(encoding="utf-8").rstrip("%"))


def _helper_table_b_differentials_excluding_cash_transfer() -> list[
    tuple[float, float]
]:
    """(coefficient, p-value) for both differential rows, all columns but cash transfer."""
    return [
        (stat("b", "coeff", column, row), stat("b", "pval", column, row))
        for row in [
            TABLE_B_TEMPERATURE_DIFFERENTIAL_ROW,
            TABLE_B_WETBULB_DIFFERENTIAL_ROW,
        ]
        for column in TABLE_B_COLUMNS
        if column != TABLE_B_CASH_TRANSFER_COLUMN
    ]


def _helper_table_b_binary_shock_differentials(row: int) -> list[float]:
    return [stat("b", "coeff", column, row) for column in TABLE_B_BINARY_SHOCK_COLUMNS]


# ---------------------------------------------------------------------------
# Table A: heat main effects
# ---------------------------------------------------------------------------


def table_a_ci95_lower_7day_temperature() -> float:
    column = TABLE_A_7DAY_TEMPERATURE_COLUMN
    return stat("a", "coeff", column, 1) - 1.96 * stat("a", "se", column, 1)


def table_a_ci95_upper_7day_temperature() -> float:
    column = TABLE_A_7DAY_TEMPERATURE_COLUMN
    return stat("a", "coeff", column, 1) + 1.96 * stat("a", "se", column, 1)


def table_a_count_heat_measures() -> int:
    return len(TABLE_A_HEAT_MEASURE_COLUMNS)


def table_a_count_insignificant_heat_measures() -> int:
    """Heat measures whose coefficient is not significant at the 10% level."""
    return sum(
        stat("a", "pval", column, 1) >= 0.10 for column in TABLE_A_HEAT_MEASURE_COLUMNS
    )


# ---------------------------------------------------------------------------
# Table B: heat x stressor differentials
# ---------------------------------------------------------------------------


def table_b_count_differentials_excluding_cash_transfer() -> int:
    return len(_helper_table_b_differentials_excluding_cash_transfer())


def table_b_count_differentials_significant_5pct_excluding_cash_transfer() -> int:
    return sum(
        p < 0.05 for _, p in _helper_table_b_differentials_excluding_cash_transfer()
    )


def table_b_count_differentials_significant_10pct_only_excluding_cash_transfer() -> int:
    """Significant at 10% but not at 5%."""
    return sum(
        0.05 <= p < 0.10
        for _, p in _helper_table_b_differentials_excluding_cash_transfer()
    )


def table_b_min_differential_temperature_binary_shocks() -> float:
    return min(
        _helper_table_b_binary_shock_differentials(TABLE_B_TEMPERATURE_DIFFERENTIAL_ROW)
    )


def table_b_max_differential_temperature_binary_shocks() -> float:
    return max(
        _helper_table_b_binary_shock_differentials(TABLE_B_TEMPERATURE_DIFFERENTIAL_ROW)
    )


def table_b_min_differential_wetbulb_binary_shocks() -> float:
    return min(
        _helper_table_b_binary_shock_differentials(TABLE_B_WETBULB_DIFFERENTIAL_ROW)
    )


def table_b_max_differential_wetbulb_binary_shocks() -> float:
    return max(
        _helper_table_b_binary_shock_differentials(TABLE_B_WETBULB_DIFFERENTIAL_ROW)
    )


def table_b_min_warm_week_effect_binary_shocks() -> float:
    """CES-D gap (SD) between the 90th and 10th percentile week, smallest binary differential."""
    return (
        TEMPERATURE_P90_P10_GAP * table_b_min_differential_temperature_binary_shocks()
    )


def table_b_max_warm_week_effect_binary_shocks() -> float:
    """CES-D gap (SD) between the 90th and 10th percentile week, largest binary differential."""
    return (
        TEMPERATURE_P90_P10_GAP * table_b_max_differential_temperature_binary_shocks()
    )


def table_b_percent_stressed_palm() -> float:
    return 100 * stat("b", "stat", TABLE_B_PALM_COLUMN, TABLE_B_STRESSED_PROPORTION_ROW)


def table_b_population_effect_palm() -> float:
    """Palm differential spread over the whole population, assuming zero effect elsewhere."""
    return stat(
        "b", "coeff", TABLE_B_PALM_COLUMN, TABLE_B_TEMPERATURE_DIFFERENTIAL_ROW
    ) * stat("b", "stat", TABLE_B_PALM_COLUMN, TABLE_B_STRESSED_PROPORTION_ROW)


# ---------------------------------------------------------------------------
# Table C: balance (column 4 = urban vehicle owners after the cut; row 8 = transport)
# ---------------------------------------------------------------------------


def table_c_percent_change_transport_vehicle_owners() -> float:
    """Post-cut change in transport spending relative to the sample mean."""
    return 100 * stat("c", "coeff", 4, 8) / stat("c", "mean", 1, 8)


def table_c_percent_gap_fuel_price_vs_transport() -> float:
    return (
        FUEL_PRICE_INCREASE_PERCENT - table_c_percent_change_transport_vehicle_owners()
    )


# ---------------------------------------------------------------------------
# Table D: palm farmer definitions
# ---------------------------------------------------------------------------


def table_d_min_differential() -> float:
    return min(stat("d", "coeff", column, 1) for column in [1, 2, 3])


def table_d_max_differential() -> float:
    return max(stat("d", "coeff", column, 1) for column in [1, 2, 3])


# ---------------------------------------------------------------------------
# Table E: job loss windows (columns 1-3 up to nine months, 4-6 a year or more)
# ---------------------------------------------------------------------------


def table_e_min_differential_up_to_9_months() -> float:
    return min(stat("e", "coeff", column, 1) for column in [1, 2, 3])


def table_e_max_differential_up_to_9_months() -> float:
    return max(stat("e", "coeff", column, 1) for column in [1, 2, 3])


def table_e_min_differential_1_year_or_more() -> float:
    return min(stat("e", "coeff", column, 1) for column in [4, 5, 6])


def table_e_max_differential_1_year_or_more() -> float:
    return max(stat("e", "coeff", column, 1) for column in [4, 5, 6])


# ---------------------------------------------------------------------------
# Table H: summary statistics (shares shown as percentages in the text)
# ---------------------------------------------------------------------------


def table_h_percent_depressed() -> float:
    return 100 * stat("h", "mean", 1, 2)


def table_h_percent_job_loss() -> float:
    return 100 * stat("h", "mean", 1, 7)


def table_h_percent_palm_farmers() -> float:
    return 100 * stat("h", "mean", 1, 8)


def table_h_percent_urban_vehicle_owners() -> float:
    return 100 * stat("h", "mean", 1, 9)


# ---------------------------------------------------------------------------
# Table I: stressor effects on household income (row 2 = DV mean)
# ---------------------------------------------------------------------------


def table_i_abs_income_loss_palm() -> float:
    return abs(stat("i", "coeff", 1, 1))


def table_i_abs_income_loss_job_loss() -> float:
    return abs(stat("i", "coeff", 2, 1))


def table_i_percent_of_mean_income_loss_palm() -> float:
    return 100 * table_i_abs_income_loss_palm() / stat("i", "stat", 1, 2)


def table_i_percent_of_mean_income_loss_job_loss() -> float:
    return 100 * table_i_abs_income_loss_job_loss() / stat("i", "stat", 2, 2)


# ---------------------------------------------------------------------------
# Table J: palm smallholders (column 2 self-employed, column 3 wage workers)
# ---------------------------------------------------------------------------


def table_j_percent_difference_self_employed_vs_wage() -> float:
    return 100 * (stat("j", "coeff", 2, 1) / stat("j", "coeff", 3, 1) - 1)


# ---------------------------------------------------------------------------
# Table N: sleep (column 1 urban vehicle owners; coefficients in hours)
# ---------------------------------------------------------------------------


def table_n_minutes_sleep_lost_vehicle_owners() -> float:
    return -60 * stat("n", "coeff", 1, 1)


DERIVED_STATS = [
    table_a_ci95_lower_7day_temperature,
    table_a_ci95_upper_7day_temperature,
    table_a_count_heat_measures,
    table_a_count_insignificant_heat_measures,
    table_b_count_differentials_excluding_cash_transfer,
    table_b_count_differentials_significant_5pct_excluding_cash_transfer,
    table_b_count_differentials_significant_10pct_only_excluding_cash_transfer,
    table_b_min_differential_temperature_binary_shocks,
    table_b_max_differential_temperature_binary_shocks,
    table_b_min_differential_wetbulb_binary_shocks,
    table_b_max_differential_wetbulb_binary_shocks,
    table_b_min_warm_week_effect_binary_shocks,
    table_b_max_warm_week_effect_binary_shocks,
    table_b_percent_stressed_palm,
    table_b_population_effect_palm,
    table_c_percent_change_transport_vehicle_owners,
    table_c_percent_gap_fuel_price_vs_transport,
    table_d_min_differential,
    table_d_max_differential,
    table_e_min_differential_up_to_9_months,
    table_e_max_differential_up_to_9_months,
    table_e_min_differential_1_year_or_more,
    table_e_max_differential_1_year_or_more,
    table_h_percent_depressed,
    table_h_percent_job_loss,
    table_h_percent_palm_farmers,
    table_h_percent_urban_vehicle_owners,
    table_i_abs_income_loss_palm,
    table_i_abs_income_loss_job_loss,
    table_i_percent_of_mean_income_loss_palm,
    table_i_percent_of_mean_income_loss_job_loss,
    table_j_percent_difference_self_employed_vs_wage,
    table_n_minutes_sleep_lost_vehicle_owners,
]


def main() -> None:
    DERIVED_OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale_file in DERIVED_OUTPUT.glob("*.tex"):
        stale_file.unlink()
    for derived_stat in DERIVED_STATS:
        (DERIVED_OUTPUT / f"{derived_stat.__name__}.tex").write_text(
            format_stat(derived_stat()), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
