from dataclasses import replace

import pandas as pd

from library.render import make_regression_table, render_table_to_latex, RegressionSpec
from library.specs import JOB_LOSS_MAIN, CONTROLS, FE_NO_WAVE
from library.config import TABLE_OUTPUT
from analysis.tables_v2.table_b_temperature_and_shock_effects import TABLE_SPECS

# Should contain vehicle fuel price for urban vehicle groups or household earnings for palm shock / job loss groups.
# vehicle_fuel_quantity_liters
transportation_spending_variable = "expenditure_transport_fuel_total_mo_real_usd"
vehicle_fuel_price_variable = "vehicle_fuel_price_per_litre"
vehicle_fuel_quantity = "vehicle_fuel_quantity_liters"
vehicle_fuel_extensive_margin = "vehicle_fuel_extensive_margin"

earnings_variable = "job_earnings_hh_real_usd"
outcome_dict = {
    "palm_farmer_hh_ifls4": earnings_variable,
    "palm_price_gap_z": earnings_variable,
    "urban_vehicle_hh_ifls4": transportation_spending_variable,
    "urban_vehicle_transfer_nonrecipient_ifls4": vehicle_fuel_price_variable,
    "urban_vehicle_transfer_recipient_ifls4": vehicle_fuel_price_variable,
    JOB_LOSS_MAIN: earnings_variable,
}
ECONOMIC_OUTCOMES = tuple(dict.fromkeys(outcome_dict.values()))


def winsorized(series: pd.Series) -> pd.Series:
    # ponytail: upper-tail cap only; use two-sided winsorization if the table spec asks.
    return series.clip(upper=series.quantile(0.95))


def make_specs():
    """
    Make regression specs following the order and structure of Table B
    """
    economic_specs = []
    for spec_data in TABLE_SPECS:
        group = spec_data["group"]
        post = spec_data["post"]
        assert isinstance(group, str)
        outcome = outcome_dict[group]
        old_spec = spec_data["spec"]
        assert isinstance(old_spec, RegressionSpec)
        df = old_spec.df.copy()  # ty:ignore[unresolved-attribute]
        df[outcome] = winsorized(df[outcome])
        dv_mean = df[outcome].mean()
        # If jobloss spec, then no post so handle that
        rhs = group if post is None else f"{group} * {post}"
        effect_term = group if post is None else f"{group}:{post}"
        post_term = post if post is not None else "post"
        spec = replace(
            old_spec,
            df=df,
            formula=f"{outcome} ~ {rhs} + {CONTROLS} | {FE_NO_WAVE}",
            show_terms=frozenset([effect_term]),
        )
        economic_specs.append(
            (
                spec,
                spec_data["label"],
                {effect_term: "Shock effect", post_term: "post"},
                "-" if pd.isna(dv_mean) else f"{dv_mean:.2f}",
            )
        )
    order = [0, 1, 2, 6, 3]
    reordered_specs = [economic_specs[i] for i in order]
    return reordered_specs


def make_table():
    specs = make_specs()
    table = make_regression_table(
        [spec for spec, _, _, _ in specs],
        titles=[label for _, label, _, _ in specs],
        rename=[rename for _, _, rename, _ in specs],
        keep=["Shock effect"],
        exact_match=True,
        custom_model_stats={"DV mean": [dv_mean for _, _, _, dv_mean in specs]},
    )
    render_table_to_latex(table, TABLE_OUTPUT / "table_i_economics.tex")


if __name__ == "__main__":
    make_table()
