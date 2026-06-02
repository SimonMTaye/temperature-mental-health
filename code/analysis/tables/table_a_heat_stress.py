"""
Build Table A: The main result table

We show that increased temperature is associated for those already under stress

We consider the following stressors:
  - Job loss within 12 months
  - Palm Shock, defined as a household working in palm agriculture that experiences the price shock
  - Fuel Shock: defined as a household that relies a lot on fuel and experiences the price shock due to subsidy cuts
"""

from pathlib import Path

import maketables as mt
import pandas as pd
import pyfixest as pf


from analysis.lib.input import drop_singleton_clusters, keep_nonmissing_rows

PROJECT = Path(__file__).resolve().parents[3]
ANALYSIS_INPUT = PROJECT / "data" / "generated" / "30_analysis_table_input.parquet"
OUTPUT = PROJECT / "output" / "tables"

# Some shared constants for this table
CONTROLS = "age + female + edu_yrs + married + widowed"
FE_DEFAULT = "month + year + wave + kabupaten_code"
CLUSTER = "kabupaten_code"


def make_pretty_table(
    labelled_models: dict[str, object],
    drop_list: list[str],
    custom_model_stats: dict[str, list[str]] | None = None,
) -> mt.ETable:
    """
    Build Formatted Latex Table for Table A
    """
    labels = {
        "cesd_z": "CES-D total (z-score)",
        "stressor_variable": "Stressor",
        "heat_c_dev": "Heat",
    }
    model_heads = list(labelled_models.keys())
    models = list(labelled_models.values())
    return mt.ETable(
        models,
        model_heads=model_heads,
        labels=labels,
        drop=drop_list,
        show_fe=False,
        custom_model_stats=custom_model_stats,
    )


def run_stressor_model(df, stressor: str, controls: str, fe: str, cluster: str):
    """
    Run a single model with the interaction between heat and a given stressor.

    Generates "stressor" variable that based on the passed parameter so
    they can appear in the same row when generating a table

    Params:
    - df: the input dataframe with all variables needed for the model
    - stressor: the name of the stressor variable to interact with heat
    - controls: the control variables to include in the model
    - fe: the fixed effects to include in the model
    - cluster: the variable to cluster standard errors by
    """
    df: pd.DataFrame = df[df[stressor].notna()].copy()
    df["stressor_variable"] = df[stressor]
    formula = f"cesd_z ~ heat_c_dev*stressor_variable + {controls} | {fe}"
    model = pf.feols(formula, data=df, vcov={"CRV1": cluster})
    return model


def calculate_total_effect(model):
    """
    Calculate the total effect of heat on the stressed group by adding
    row coefficient and interaction coefficient, and calculate the standard error using the variance-covariance matrix

    Params:
    - model: the fitted regression model with the interaction term
    - stressor_name: the name of the stressor variable used in the interaction term
    """
    coefs = model.coef()
    vcov = pd.DataFrame(model._vcov, index=coefs.index, columns=coefs.index)

    heat = "heat_c_dev"
    interaction = "heat_c_dev:stressor_variable"

    estimate = coefs[heat] + coefs[interaction]
    se = (
        vcov.loc[heat, heat]
        + vcov.loc[interaction, interaction]
        + 2 * vcov.loc[heat, interaction]
    ) ** 0.5

    return estimate, se


def main_table(
    path_to_data: str | Path = ANALYSIS_INPUT,
    controls: str = CONTROLS,
    fe: str = FE_DEFAULT,
    cluster: str = CLUSTER,
) -> mt.ETable:
    """
    Build the main Table A showing the interaction between heat and stressors on CES-D total.
    """
    df = pd.read_parquet(path_to_data)
    # Drop some rows that correspond to districts (kabupaten) observed only in one wave
    df = drop_singleton_clusters(df)
    df = keep_nonmissing_rows(df, ["cesd_z", "heat_c_dev"])

    no_interaction_formula = f"cesd_z ~ heat_c_dev + {controls} | {fe}"
    no_interaction_model = pf.feols(
        no_interaction_formula, data=df, vcov={"CRV1": cluster}
    )

    stressors = ["job_loss_within_yr", "palm_shock"]
    stressor_models = [
        run_stressor_model(df, stressor, controls, fe, cluster)
        for stressor in stressors
    ]
    # Run fuel shock on IFLS5 only since the fuel shock variable is only defined in that wave
    stressor_models.append(
        run_stressor_model(df[df.wave == "IFLS5"], "fuel_shock", controls, fe, cluster)
    )
    # Calculate the 'total' effect of heat on the stressed group by adding
    # row coefficient and interaction coefficient, and calculate the standard error using the variance-covariance matrix
    total_effects = [calculate_total_effect(model) for model in stressor_models]
    total_coeffs = ["---"] + [f"{effect[0]:.3f}" for effect in total_effects]
    total_ses = ["---"] + [f"({effect[1]:.3f})" for effect in total_effects]
    wave_fe = ["Yes"] * 3 + ["IFLS5 Only"]

    models = [no_interaction_model, *stressor_models]
    labelled_models = {
        "Pooled (no interactions)": models[0],
        "Job Loss": models[1],
        "Palm Shock": models[2],
        "Fuel Cut": models[3],
    }
    control_list = controls.split(" + ")
    return make_pretty_table(
        labelled_models,
        drop_list=control_list,
        custom_model_stats={
            "Marginal Effect of Heat": total_coeffs,
            " ": total_ses,
            "Wave FE": wave_fe,
        },
    )


def main():
    table = main_table()
    table_path = str(OUTPUT / "table_a_heat_stress.tex")
    table.save("tex", table_path, replace=True)
    print(f"Saved Table A to {table_path}")


if __name__ == "__main__":
    main()
