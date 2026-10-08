"""Rescale heat effect sizes using temperature variation left after fixed effects.

Run from the repo root to print the numbers:

    uv run python code/analysis/updates/residual_temperature_magnitudes.py

Responds to reviewer comment 6 ("Magnitude of the heat variation used for
interpretation"): report the residual SD of temperature after the fixed effects
and rescale the magnitude discussion accordingly.

Old calculation (paper text, Table B paragraph)
    1. Raw spread: across the full sample, the 90th-percentile week of
       `tmean_7d` minus the 10th-percentile week = 4.5 °C.
    2. Coefficients: Table B binary-group interaction coefficients range
       from 0.044 to 0.090 SD per °C.
    3. Multiply: 0.044 x 4.5 ~ 0.20 SD and 0.090 x 4.5 ~ 0.41 SD.
    4. Compare: 30-100% of CBT / pharmacotherapy effects.
    Problem: the 4.5 °C includes variation the regression never uses (some
    kecamatans are hotter than others, some month-years are hotter than
    others). The fixed effects absorb that; the coefficient is identified only
    off what is left.

Updated calculation (this script)
    1. Regress temperature on everything else in the headline model:
       `heat ~ controls | month*year + kecamatan`, same controls, FEs and
       sample as the main-effect regression. A wave-FE version is also
       printed; it is identical because waves never share a calendar year.
    2. Keep the residuals: how much hotter or cooler that respondent's week was
       than usual for that kecamatan and month-year. By Frisch-Waugh-Lovell,
       this is exactly the variation the coefficient comes from.
    3. Measure their spread the same way: residual p90 - p10 (~1.1 °C, vs
       4.5 °C raw).
    4. Multiply again: coefficient x residual p90 - p10.
    As in the paper, one range is computed on the full sample with the
    headline FEs (kecamatan, month-by-year) and applied to every column: the
    goal is to contextualize weather variation across all of Indonesia, not
    within each column's sample. Wet-bulb gets its own full-sample range.
    The coefficients do not change; only the size of a "typical shock" does.
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from analysis.tables_v2.table_b_temperature_and_shock_effects import (
    TABLE_SPECS,
    regression_runner,
    wetbulb_specs,
)
from library.caching import run_regression_with_caching
from library.specs import (
    CONTROLS,
    FE_NO_WAVE,
    FE_WAVE,
    MAIN_TEMP_MEASURE,
    analysis_df,
    temperature_spec,
)

OUTCOME = "cesd_z"
WETBULB_MEASURE = "wetbulb_7d"

# Columns the paper quotes as "binary indicators" (0.044-0.090 SD per °C):
# excludes the continuous palm price gap and the cash-transfer recipients.
PAPER_RANGE_COLUMNS = [
    "Palm Shock",
    "Palm Shocks Panel",
    "Urban Vehicle Owners",
    "Urban Vehicle Owners No Cash Transfer",
    "Job Loss",
]

MAIN_EFFECT_SPECS = [
    {
        "spec": temperature_spec,
        "group": None,
        "post": None,
        "heat": MAIN_TEMP_MEASURE,
        "label": "Main effect",
    },
]


def full_sample_heat_ranges(heat: str, fixed_effects: str) -> dict[str, float]:
    """Raw and residual p90 - p10 of heat on the full headline-spec sample.

    Identification assumes daily weather deviations around the kecamatan and
    month-by-year norms are as good as random conditional on demographics, so
    the residuals are the variation that identifies the heat coefficients.
    """
    sample = analysis_df.dropna(subset=[OUTCOME, heat])
    model = pf.feols(f"{heat} ~ {CONTROLS} | {fixed_effects}", data=sample)
    raw, residual = model._data[heat].to_numpy(), model.resid()
    return {
        "raw_sd": raw.std(),
        "raw_p90_p10": np.quantile(raw, 0.9) - np.quantile(raw, 0.1),
        "resid_sd": residual.std(),
        "resid_p90_p10": np.quantile(residual, 0.9) - np.quantile(residual, 0.1),
    }


def heat_coefficient(model, spec_data: dict) -> float:
    if spec_data["group"] is None:
        return float(model.coef_table.loc[spec_data["heat"], "b"])
    term = "group:post:heat" if spec_data["post"] is not None else "group:heat"
    return float(model.coef_table.loc[term, "b"])


def magnitude_rows(specs: list[dict], models: list, ranges: dict) -> pd.DataFrame:
    rows = []
    for spec_data, model in zip(specs, models):
        coefficient = heat_coefficient(model, spec_data)
        heat_range = ranges[spec_data["heat"]]
        rows.append(
            {
                "column": spec_data["label"]
                .replace(r"\shortstack{", "")
                .replace(r"\\", " ")
                .replace("}", ""),
                "heat": spec_data["heat"],
                "coef_per_degree": coefficient,
                "old_effect": coefficient * heat_range["raw_p90_p10"],
                "new_effect": coefficient * heat_range["resid_p90_p10"],
            }
        )
    return pd.DataFrame(rows)


def print_paper_range(results: pd.DataFrame, heat: str) -> None:
    subset = results[
        results["column"].isin(PAPER_RANGE_COLUMNS) & results["heat"].eq(heat)
    ]
    print(
        f"{heat}: per-degree {subset['coef_per_degree'].min():.3f} to "
        f"{subset['coef_per_degree'].max():.3f} SD | "
        f"old p10-p90 effect {subset['old_effect'].min():.2f} to "
        f"{subset['old_effect'].max():.2f} SD | "
        f"new p10-p90 effect {subset['new_effect'].min():.3f} to "
        f"{subset['new_effect'].max():.3f} SD"
    )


def main() -> None:
    main_effect_models = [
        run_regression_with_caching(spec_data["spec"])
        for spec_data in MAIN_EFFECT_SPECS
    ]
    temperature_models = regression_runner(TABLE_SPECS)
    wetbulb_models = regression_runner(wetbulb_specs())

    # The outcome regressions use FE_NO_WAVE. Waves never share a calendar
    # year (IFLS4: 2007-08, IFLS5: 2014-15), so year FEs already absorb wave
    # and both residualizations should give the same ranges.
    for fe_label, fixed_effects in [("no wave FE", FE_NO_WAVE), ("wave FE", FE_WAVE)]:
        ranges = {
            heat: full_sample_heat_ranges(heat, fixed_effects)
            for heat in [MAIN_TEMP_MEASURE, WETBULB_MEASURE]
        }
        results = pd.concat(
            [
                magnitude_rows(MAIN_EFFECT_SPECS, main_effect_models, ranges),
                magnitude_rows(TABLE_SPECS, temperature_models, ranges),
                magnitude_rows(wetbulb_specs(), wetbulb_models, ranges),
            ],
            ignore_index=True,
        )

        print(f"\n===== Residualized on: {fixed_effects} ({fe_label}) =====")
        with pd.option_context("display.width", 200, "display.precision", 3):
            print("Full-sample heat variation (°C):")
            print(pd.DataFrame(ranges).T.to_string())
            print("\nEffect of a p10 -> p90 week (CES-D SD):")
            print(results.to_string(index=False))
        print("\nRange quoted in the paper (binary indicators, excl. cash recipients):")
        print_paper_range(results, MAIN_TEMP_MEASURE)
        print_paper_range(results, WETBULB_MEASURE)


if __name__ == "__main__":
    main()
