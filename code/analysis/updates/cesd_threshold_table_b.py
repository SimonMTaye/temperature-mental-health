"""Re-run Table B with binary CES-D-10 cutoffs as the outcome instead of cesd_z.

Run from the repo root:

    uv run python code/analysis/updates/cesd_threshold_table_b.py

Writes one table per cutoff to output/tables/table_b_cesd_ge{cutoff}.tex and
prints the differential-heat coefficients for a quick read.

Cutoffs (IFLS uses the 10-item Andresen CES-D, raw score 0-30):
    >= 10  Andresen et al. (1994) screening cutoff; the standard in IFLS5 papers
           (which report ~23% prevalence in IFLS5, matching our sample). It has
           not been validated for Indonesia.
    >= 8   Lenient cutoff tested by Bjorgvinsson et al. (2013): higher
           sensitivity, lower specificity.
    >= 12  Stricter cutoff that was optimal in Cheng & Chan (2005, older Chinese
           adults) and close to the 11-13 optima in Baron et al. (2017, South
           Africa).
Sample prevalence (IFLS4 / IFLS5): >=8 13% / 34%, >=10 7% / 23%, >=12 4% / 16%.

Coefficients are linear-probability effects: change in the probability of
scoring at or above the cutoff per °C. Identification is the same as Table B:
kecamatan and month-by-year fixed effects, so heat effects come from weekly
weather deviations around local and calendar norms.
"""

from dataclasses import replace

import pandas as pd
from analysis.tables_v2.table_b_temperature_and_shock_effects import (
    TABLE_SPECS,
    TABLE_TEMPLATE,
    coefficient_stats,
    panel_rows,
    regression_runner,
    stressed_group_proportion_row,
    wetbulb_specs,
)
from library.config import TABLE_OUTPUT
from library.specs import update_formula_search_replace
from library.table_builder import make_row

CUTOFFS = [8, 10, 12]


def cutoff_specs(specs: list[dict], cutoff: int) -> list[dict]:
    """Swap the cesd_z outcome for 1{cesd_raw >= cutoff} in each spec."""
    outcome = f"cesd_ge{cutoff}"
    updated_specs = []
    for spec_data in specs:
        spec = update_formula_search_replace(spec_data["spec"], "cesd_z", outcome)
        df = spec.df.assign(**{outcome: spec.df["cesd_raw"].ge(cutoff).astype(int)})
        updated_specs.append({**spec_data, "spec": replace(spec, df=df)})
    return updated_specs


def outcome_mean_row(specs: list[dict], cutoff: int) -> str:
    return make_row(
        "Outcome Mean",
        [f"{spec_data['spec'].df[f'cesd_ge{cutoff}'].mean():.3f}" for spec_data in specs],
    )


def build_table(cutoff: int) -> tuple[str, list, list]:
    temperature_specs = cutoff_specs(TABLE_SPECS, cutoff)
    temperature_models = regression_runner(temperature_specs)
    wetbulb_models = regression_runner(wetbulb_specs(temperature_specs))
    observations_row = make_row(
        "Observations", [f"{int(model.stats['N']):,}" for model in temperature_models]
    )
    table = (
        TABLE_TEMPLATE.replace("CES-D z-score", rf"CES-D $\geq$ {cutoff}")
        .replace(
            "{temperature_panel}",
            panel_rows("7-day mean Temperature", temperature_models),
        )
        .replace("{wetbulb_panel}", panel_rows("7-day mean Wet-bulb", wetbulb_models))
        .replace("{observations_row}", observations_row)
        .replace(
            "{group_mean_row}",
            stressed_group_proportion_row(temperature_specs)
            + "\n"
            + outcome_mean_row(temperature_specs, cutoff),
        )
    )
    return table, temperature_models, wetbulb_models


def differential_summary(cutoff: int, heat: str, models: list) -> pd.DataFrame:
    rows = []
    for spec_data, model in zip(TABLE_SPECS, models):
        term = "group:post:heat" if spec_data["post"] is not None else "group:heat"
        coefficient, standard_error, p_value = coefficient_stats(model, term)
        rows.append(
            {
                "cutoff": cutoff,
                "heat": heat,
                "column": spec_data["label"]
                .replace(r"\shortstack{", "")
                .replace(r"\\", " ")
                .replace("}", ""),
                "b": coefficient,
                "se": standard_error,
                "p": p_value,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for cutoff in CUTOFFS:
        table, temperature_models, wetbulb_models = build_table(cutoff)
        (TABLE_OUTPUT / f"table_b_cesd_ge{cutoff}.tex").write_text(
            table, encoding="utf-8"
        )
        summaries.append(differential_summary(cutoff, "tmean_7d", temperature_models))
        summaries.append(differential_summary(cutoff, "wetbulb_7d", wetbulb_models))

    with pd.option_context("display.width", 200, "display.precision", 4):
        print("Differential impact of heat (LPM, per °C):")
        print(pd.concat(summaries, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
