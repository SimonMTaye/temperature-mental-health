from dataclasses import replace
from library.config import TABLE_OUTPUT
from library.specs import (
    fuel_shock_urban_vehicle,
    MAIN_TEMP_MEASURE,
    update_formula_search_replace,
    analysis_df,
)
from library.render import make_shock_regression_table_trimmed, render_table_to_latex


def make_table() -> None:
    wave5 = (
        analysis_df.copy()
        .query("wave == 'IFLS5'")
        .assign(
            urban_vehicle_transfer_recipient_ifls4=lambda df: (
                df["urban_vehicle_hh_ifls4"] & df["cash_transfer_recipient"]
            ).astype("Int32"),
            urban_vehicle_transfer_nonrecipient_ifls4=lambda df: (
                df["urban_vehicle_hh_ifls4"] & ~df["cash_transfer_recipient"]
            ).astype("Int32"),
        )
    )
    groups = [
        ("Urban Vehicle Owners", "urban_vehicle_hh_ifls4"),
        (
            r"\shortstack{Urban Vehicle Owners\\Cash Transfer}",
            "urban_vehicle_transfer_recipient_ifls4",
        ),
        (
            r"\shortstack{Urban Vehicle Owners\\No Cash Transfer}",
            "urban_vehicle_transfer_nonrecipient_ifls4",
        ),
    ]

    specs = [
        replace(
            update_formula_search_replace(
                fuel_shock_urban_vehicle,
                "urban_vehicle_hh_ifls4",
                term,
            ),
            df=wave5,
        )
        for _, term in groups
    ]

    table = make_shock_regression_table_trimmed(
        specs,
        group=[term for _, term in groups],
        post="post_subsidy",
        temperature=MAIN_TEMP_MEASURE,
        titles=[label for label, _ in groups],
    )
    render_table_to_latex(table, TABLE_OUTPUT / "table_g_fuel_shock_alt_definition.tex")


if __name__ == "__main__":
    make_table()
