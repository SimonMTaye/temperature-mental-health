from library.config import TABLE_OUTPUT
from library.specs import (
    palm_shock,
    analysis_df,
    update_formula_search_replace,
    MAIN_TEMP_MEASURE,
)
from library.render import make_shock_regression_table_trimmed, render_table_to_latex
from dataclasses import replace


def make_table() -> None:
    wave5_palm = (
        analysis_df.copy()
        .query("wave == 'IFLS5'")
        .filter(items=["pidlink", "palm_farmer_hh"])
        .add_suffix("_ifls5")
        .rename(columns={"pidlink_ifls5": "pidlink"})
        .assign(
            palm_farmer_hh_ifls5=lambda df: (
                df["palm_farmer_hh_ifls5"].astype("Int32").fillna(0)
            )
        )
    )
    wave4 = (
        analysis_df.copy()
        .query("wave == 'IFLS4'")
        .filter(items=["pidlink", "farmer_hh", "palm_region"])
        .assign(
            farmer_non_palm_ifls4=lambda df: (
                df["farmer_hh"] & ~df["palm_region"]
            ).astype("Int32"),
        )
        .filter(items=["pidlink", "farmer_non_palm_ifls4"])
    )

    palm_df = (
        analysis_df.copy()
        .merge(wave4, on="pidlink", how="left", validate="m:1")
        .merge(wave5_palm, on="pidlink", how="left", validate="m:1")
        .assign(
            palm_farmer_both_waves=lambda df: (
                df["palm_farmer_hh_ifls5"] & df["palm_farmer_hh_ifls4"]
            ).astype("Int32"),
        )
    )

    groups = [
        ("Main Definition", "palm_farmer_hh_ifls4"),
        (r"Both IFLS4\&5", "palm_farmer_both_waves"),
        ("IFLS5", "palm_farmer_hh_ifls5"),
        # (r"\shortstack{Farmer non-palm\\region}", "farmer_non_palm_ifls4"),
    ]
    specs = [
        replace(
            update_formula_search_replace(
                palm_shock,
                "palm_farmer_hh_ifls4",
                term,
            ),
            df=palm_df,
        )
        for _, term in groups
    ]

    table = make_shock_regression_table_trimmed(
        specs,
        group=[term for _, term in groups],
        post="ifls5",
        temperature=MAIN_TEMP_MEASURE,
        titles=[label for label, _ in groups],
    )
    render_table_to_latex(
        table, TABLE_OUTPUT / "table_d_palm_farmer_alt_definition.tex"
    )


if __name__ == "__main__":
    make_table()
