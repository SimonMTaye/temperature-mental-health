from pathlib import Path

import pandas as pd
from render import RegressionSpec, search_replace_term

PROJECT = Path(__file__).parent.parent.parent
ANALYSIS_INPUT = PROJECT / "data" / "generated" / "30_analysis_table_input.parquet"
CONTROLS = "age + female + edu_yrs + married + widowed"
FE_WAVE = "month+year+ifls5+gadm_fullcode"
FE_NO_WAVE = "month+year+gadm_fullcode"
MAIN_TEMP_MEASURE = "tmean_c_dev"


def update_formula_search_replace(
    spec: RegressionSpec,
    old_term: str,
    new_term: str,
) -> RegressionSpec:
    """Return a spec with formula and displayed terms updated by whole-token replacement."""
    formula, found = search_replace_term(spec.formula, old_term, new_term)
    if not found:
        raise ValueError(f"{old_term!r} not found in formula: {spec.formula}")

    show_terms = None
    if spec.show_terms is not None:
        show_terms = frozenset(
            search_replace_term(term, old_term, new_term)[0] for term in spec.show_terms
        )

    return RegressionSpec(
        title=spec.title,
        formula=formula,
        df=spec.df,
        tags=spec.tags,
        show_terms=show_terms,
    )


def shock_triple_diff(outcome: str, measure: str, shock: str) -> str:
    """Return a formula for a standard triple-difference specification of shock effects."""
    return f"{outcome} ~ {measure} * ifls5 * {shock} + {CONTROLS} | {FE_WAVE}"


def shock_triple_diff_panel(outcome: str, heat: str, group: str, post: str) -> str:
    """Return a formula for a triple-difference specification of shock effects in the IFLS5 panel."""
    return f"""
      {outcome} ~ {heat} + {group}:{post}:{heat} + {group}:{heat} + {post}:{heat} + {group}:{post} + {CONTROLS} | {FE_NO_WAVE}+pidlink+{post}
    """


analysis_df = pd.read_parquet(ANALYSIS_INPUT)
analysis_df["ifls5"] = analysis_df["wave"] == "IFLS5"
wave5_df = analysis_df[analysis_df["ifls5"]].copy()


temperature_spec = RegressionSpec(
    title="Main Effects of Temperature",
    formula=f"cesd_z ~ tmean_c_dev + {CONTROLS} | {FE_WAVE}",
    df=analysis_df,
    tags=frozenset(["mean-daily-temp", "temperature-effect"]),
    show_terms=frozenset(["tmean_c_dev"]),
)

palm_shock = RegressionSpec(
    title="Palm Shock",
    formula=f"cesd_z ~ palm_farmer_hh_ifls4 * ifls5 * tmean_c_dev + {CONTROLS} | {FE_NO_WAVE}",
    df=analysis_df,
    tags=frozenset(["palm-shock", "mean-daily-temp"]),
    show_terms=frozenset(["palm_farmer_hh_ifls4:ifls5:tmean_c_dev"]),
)


fuel_shock = RegressionSpec(
    title="Fuel Cut",
    formula=f"cesd_z ~ urban_vehicle_hh_ifls4 * post_subsidy * tmean_c_dev + {CONTROLS} | {FE_NO_WAVE}",
    df=wave5_df,
    tags=frozenset(["fuel-shock", "mean-daily-temp"]),
    show_terms=frozenset(["urban_vehicle_hh_ifls4:post_subsidy:tmean_c_dev"]),
)

coal_shock = RegressionSpec(
    title="Coal Shock",
    formula=f"cesd_z ~ coal_worker_hh_ifls4 * ifls5 * tmean_c_dev + {CONTROLS} | {FE_NO_WAVE}",
    df=analysis_df,
    tags=frozenset(["coal-shock", "mean-daily-temp"]),
    show_terms=frozenset(["coal_worker_hh_ifls4:ifls5:tmean_c_dev"]),
)

jobloss = RegressionSpec(
    title="Job Loss",
    formula=f"cesd_z ~ job_loss_1_yr * tmean_c_dev + {CONTROLS} | {FE_WAVE}",
    df=analysis_df,
    tags=frozenset(["job-loss", "mean-daily-temp"]),
    show_terms=frozenset(["job_loss_1_yr:tmean_c_dev"]),
)


palm_shock_panel = RegressionSpec(
    title="Palm Shocks - Panel",
    formula=shock_triple_diff_panel(
        outcome="cesd_z",
        heat="tmean_c_dev",
        post="ifls5",
        group="palm_farmer_hh_ifls4",
    ),
    df=analysis_df,
    tags=frozenset(["palm-shock", "mean-daily-temp", "panel"]),
    show_terms=frozenset(["palm_farmer_hh_ifls4:ifls5:tmean_c_dev"]),
)

fuel_shock_panel = RegressionSpec(
    title="Fuel Cut - Panel",
    formula=shock_triple_diff_panel(
        outcome="cesd_z",
        heat="tmean_c_dev",
        post="post_subsidy",
        group="urban_vehicle_hh_ifls4",
    ),
    df=analysis_df,
    tags=frozenset(["fuel-shock", "mean-daily-temp", "panel"]),
    show_terms=frozenset(["urban_vehicle_hh_ifls4:post_subsidy:tmean_c_dev"]),
)

coal_shock_panel = RegressionSpec(
    title="Coal Shock - Panel",
    formula=shock_triple_diff_panel(
        outcome="cesd_z",
        heat="tmean_c_dev",
        post="ifls5",
        group="coal_worker_hh_ifls4",
    ),
    df=analysis_df,
    tags=frozenset(["coal-shock", "mean-daily-temp", "panel"]),
    show_terms=frozenset(["coal_worker_hh_ifls4:ifls5:tmean_c_dev"]),
)
