from dataclasses import replace

from library.render import make_regression_table, render_table_to_latex, RegressionSpec
from library.specs import JOB_LOSS_MAIN, CONTROLS, FE_NO_WAVE
from library.config import TABLE_OUTPUT
from analysis.tables_v2.table_b_temperature_and_shock_effects import TABLE_SPECS

# Should contain "expenditure" for fuel share groups or "job_income_hh" for palm shock / job loss groups
outcome_dict = {
    "palm_farmer_hh_ifls4": "job_earnings_hh_real",
    "palm_price_gap_z": "job_earnings_hh_real",
    "fuel_share_ifls4": "expenditure_total_mo_real",
    "urban_vehicle_hh_ifls4": "expenditure_total_mo_real",
    JOB_LOSS_MAIN: "job_earnings_hh_real",
}


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
        df[outcome] = df[outcome] / 1_000_000
        # If jobloss spec, then no post so handle that
        rhs = group if post is None else f"{group} * {post}"
        effect_term = group if post is None else f"{group}:{post}"
        spec = replace(
            old_spec,
            df=df,
            formula=f"{outcome} ~ {rhs} + {CONTROLS} | {FE_NO_WAVE}",
            show_terms=frozenset([effect_term]),
        )
        economic_specs.append((spec, spec_data["label"], {effect_term: "Shock effect"}))
    return economic_specs


def make_table():
    specs = make_specs()
    table = make_regression_table(
        [spec for spec, _, _ in specs],
        titles=[label for _, label, _ in specs],
        rename=[rename for _, _, rename in specs],
        keep=["Shock effect"],
        exact_match=True,
    )
    render_table_to_latex(table, TABLE_OUTPUT / "table_i_economics.tex")


if __name__ == "__main__":
    make_table()
