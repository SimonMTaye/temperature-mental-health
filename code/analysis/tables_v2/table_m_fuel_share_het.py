from dataclasses import replace
from library.config import TABLE_OUTPUT
from library.specs import (
    fuel_shock_fuel_share,
    MAIN_TEMP_MEASURE,
    update_formula_search_replace,
    analysis_df,
    FUEL_SHARE_MAIN,
)
from library.render import make_shock_regression_table_trimmed, render_table_to_latex


def make_table() -> None:
    wave5 = analysis_df.copy().query("wave == 'IFLS5'")
    transfer_recipient = wave5["cash_transfer_recipient"].fillna(0).astype(bool)
    transfer_recipient_ifls4 = (
        wave5["cash_transfer_recipient_ifls4"].fillna(0).astype(bool)
    )
    wave5 = wave5.assign(
        fuel_share_recipient=(wave5["fuel_transport_share_100"] * transfer_recipient),
        fuel_share_nonrecipient=(
            wave5["fuel_transport_share_100"] * ~transfer_recipient
        ),
        fuel_share_nonrecipient_ifls4=(
            wave5[FUEL_SHARE_MAIN] * ~transfer_recipient_ifls4
        ),
        fuel_share_recipient_ifls4=(wave5[FUEL_SHARE_MAIN] * transfer_recipient_ifls4),
        fuel_share4_nonrecipient_5=(wave5[FUEL_SHARE_MAIN] * ~transfer_recipient),
        fuel_share4_recipient_5=(wave5[FUEL_SHARE_MAIN] * transfer_recipient),
    )
    groups = [
        ("Fuel Share", FUEL_SHARE_MAIN),
        (
            r"\shortstack{(IFLS5) Fuel Share\\(IFLS5) Cash Transfer Recipient}",
            "fuel_share_recipient",
        ),
        (
            r"\shortstack{(IFLS5) Fuel Share\\(IFLS5) No Cash Transfer}",
            "fuel_share_nonrecipient",
        ),
        (
            r"\shortstack{(IFLS4) Fuel Share\\(IFLS4) Cash Transfer Recipient}",
            "fuel_share_recipient_ifls4",
        ),
        (
            r"\shortstack{(IFLS4) Fuel Share\\ (IFLS4) No Cash Transfer}",
            "fuel_share_nonrecipient_ifls4",
        ),
        ###
        (
            r"\shortstack{(IFLS4) Fuel Share\\(IFLS5) No Cash Transfer}",
            "fuel_share4_nonrecipient_5",
        ),
        (
            r"\shortstack{(IFLS4) Fuel Share\\(IFLS5) Cash Transfer}",
            "fuel_share4_recipient_5",
        ),
    ]

    specs = [
        replace(
            update_formula_search_replace(
                fuel_shock_fuel_share,
                FUEL_SHARE_MAIN,
                term,
            ),
            df=wave5,
        )
        for _, term in groups
    ]

    table = make_shock_regression_table_trimmed(
        specs,
        group=[term for _, term in groups],
        post=["post_subsidy"] * len(groups),
        temperature=MAIN_TEMP_MEASURE,
        titles=[label for label, _ in groups],
    )
    render_table_to_latex(table, TABLE_OUTPUT / "table_m_fuel_share_het.tex")


if __name__ == "__main__":
    make_table()
