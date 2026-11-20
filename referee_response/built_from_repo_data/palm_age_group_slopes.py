"""Palm panel specification with heat slopes that change by baseline age group.

Run from the repo root to print the numbers:

    uv run python code/analysis/updates/palm_age_group_slopes.py

Responds to Reviewer 3, comment 1 (panel ageing): respondents are seven years
older in IFLS5, so a change in heat sensitivity with age could load on the
palm x IFLS5 x heat term. The check adds, to Table B column 2, heat slopes for
respondents aged 40-59 and 60+ at the IFLS4 interview (under 40 is the
reference) and lets those slopes change between waves. Sample, controls, fixed
effects and district clustering are the same as in Table B column 2.
"""

from dataclasses import replace

import pandas as pd
from analysis.tables_v2.table_b_temperature_and_shock_effects import (
    coefficient_stats,
    regression_runner,
    wetbulb_specs,
)
from library.specs import MAIN_TEMP_MEASURE, analysis_df, palm_shock_panel

AGE_GROUPS = {"age_40_59_ifls4": (40, 59), "age_60_plus_ifls4": (60, 200)}


def age_group_spec() -> dict:
    df = analysis_df.copy()
    baseline_age = df.loc[~df["ifls5"]].set_index("pidlink")["age"]
    df["age_ifls4"] = df["pidlink"].map(baseline_age)
    df = df.dropna(subset=["age_ifls4"])
    for column, (low, high) in AGE_GROUPS.items():
        df[column] = df["age_ifls4"].between(low, high).astype(int)
    rhs, fixed_effects = palm_shock_panel.formula.split("|")
    for column in AGE_GROUPS:
        rhs += (
            f" + {column}:{MAIN_TEMP_MEASURE} + {column}:ifls5"
            f" + {column}:ifls5:{MAIN_TEMP_MEASURE} "
        )
    return {
        "spec": replace(palm_shock_panel, formula=f"{rhs}|{fixed_effects}", df=df),
        "group": "palm_farmer_hh_ifls4",
        "post": "ifls5",
        "heat": MAIN_TEMP_MEASURE,
    }


def main() -> None:
    specs = [age_group_spec()]
    rows = []
    for heat_specs in [specs, wetbulb_specs(specs)]:
        (model,) = regression_runner(heat_specs)
        for term in ["group:post:heat", *[f"{c}:post:heat" for c in AGE_GROUPS]]:
            coefficient, standard_error, p_value = coefficient_stats(model, term)
            rows.append(
                {
                    "heat": heat_specs[0]["heat"],
                    "term": term,
                    "b": coefficient,
                    "se": standard_error,
                    "p": p_value,
                    "N": model.stats["N"],
                }
            )
    with pd.option_context("display.width", 200, "display.precision", 4):
        print("Change in heat slope between waves (SD per °C):")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
