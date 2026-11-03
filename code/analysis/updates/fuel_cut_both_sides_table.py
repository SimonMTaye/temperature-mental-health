"""Re-run Table G (fuel shock) on kecamatans interviewed both before and after the cut.

Run from the repo root:

    uv run python code/analysis/updates/fuel_cut_both_sides_table.py

Writes output/tables/table_g_fuel_shock_both_sides.tex. Each Table G group
gets two adjacent columns: the full IFLS5 sample and only kecamatans with at
least one IFLS5 interview before and one after the Nov 18, 2014 fuel subsidy
cut (any respondent, not only urban vehicle owners; see
fuel_cut_share_map.py). About a third of IFLS5 respondents and 91% of
pre-cut respondents are in these kecamatans.

With kecamatan fixed effects, the group x post x heat term can only be
identified from kecamatans observed on both sides of the cut; kecamatans
interviewed entirely on one side still shape the heat slopes and controls.
Dropping them checks that the fuel result is not driven by those one-sided
kecamatans. Identification is otherwise unchanged: kecamatan and
month-by-year fixed effects, heat effects from weekly weather deviations
around local and calendar norms.
"""

from dataclasses import replace

import pandas as pd
from analysis.tables_v2.table_b_temperature_and_shock_effects import (
    coefficient_stats,
    standardize_model_terms,
)
from library.caching import run_regression_with_caching
from library.config import TABLE_OUTPUT
from library.specs import (
    MAIN_TEMP_MEASURE,
    analysis_df,
    fuel_shock_urban_vehicle,
    update_formula_search_replace,
)
from library.table_builder import coefficient_rows, make_row

GROUPS = [
    ("Urban Vehicle Owners", "urban_vehicle_hh_ifls4"),
    ("Cash Transfer", "urban_vehicle_transfer_recipient_ifls4"),
    ("No Cash Transfer", "urban_vehicle_transfer_nonrecipient_ifls4"),
]

TABLE_TEMPLATE = r"""
\begin{tabular}{@{}lcccccc}
\toprule
 & \multicolumn{6}{c}{CES-D z-score} \\
\cmidrule(lr){2-7}
 & \multicolumn{2}{c}{Urban Vehicle Owners} & \multicolumn{2}{c}{\shortstack{Urban Vehicle Owners\\Cash Transfer}} & \multicolumn{2}{c}{\shortstack{Urban Vehicle Owners\\No Cash Transfer}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
 & Full & Both Sides & Full & Both Sides & Full & Both Sides \\
 & (1) & (2) & (3) & (4) & (5) & (6) \\
\midrule\addlinespace[2.5pt]
{coefficient_rows}
\midrule\addlinespace[2.5pt]
Kecamatan & x & x & x & x & x & x \\
Month-Year & x & x & x & x & x & x \\
\midrule\addlinespace[2.5pt]
{observations_row}
{group_mean_row}
\bottomrule
\end{tabular}
"""


def wave5_with_fuel_groups() -> pd.DataFrame:
    """IFLS5 sample with the Table G cash-transfer splits."""
    wave5 = analysis_df.copy().query("wave == 'IFLS5'")
    urban_vehicle = wave5["urban_vehicle_hh_ifls4"]
    transfer_recipient = wave5["cash_transfer_recipient"]
    return wave5.assign(
        urban_vehicle_transfer_recipient_ifls4=(
            urban_vehicle * transfer_recipient
        ).astype("Int32"),
        urban_vehicle_transfer_nonrecipient_ifls4=(
            urban_vehicle * (1 - transfer_recipient)
        ).astype("Int32"),
    )


def keep_kecamatans_on_both_sides(wave5: pd.DataFrame) -> pd.DataFrame:
    share_after = (
        wave5.groupby("gadm_fullcode")["post_subsidy"].transform("mean").astype(float)
    )
    return wave5.loc[share_after.between(0, 1, inclusive="neither")].copy()


def interleaved_specs(full: pd.DataFrame, both_sides: pd.DataFrame) -> list[dict]:
    """One full-sample and one both-sides spec per group, adjacent."""
    specs = []
    for _, group in GROUPS:
        spec = update_formula_search_replace(
            fuel_shock_urban_vehicle, "urban_vehicle_hh_ifls4", group
        )
        for df in [full, both_sides]:
            specs.append({"spec": replace(spec, df=df), "group": group})
    return specs


def run_models(specs: list[dict]) -> list:
    return [
        standardize_model_terms(
            run_regression_with_caching(spec_data["spec"]),
            {
                spec_data["group"]: "group",
                "post_subsidy": "post",
                MAIN_TEMP_MEASURE: "heat",
            },
        )
        for spec_data in specs
    ]


def build_table(specs: list[dict], models: list) -> str:
    differential_coefs, differential_ses = coefficient_rows(
        "Differential Impact of Heat",
        [coefficient_stats(model, "group:post:heat") for model in models],
    )
    heat_coefs, heat_ses = coefficient_rows(
        "Heat", [coefficient_stats(model, "heat") for model in models]
    )
    observations_row = make_row(
        "Observations", [f"{int(model.stats['N']):,}" for model in models]
    )
    group_mean_row = make_row(
        "Stressed Group Proportion",
        [
            f"{spec_data['spec'].df[spec_data['group']].mean():.3f}"
            for spec_data in specs
        ],
    )
    return (
        TABLE_TEMPLATE.replace(
            "{coefficient_rows}",
            f"{differential_coefs}\n{differential_ses}\n{heat_coefs}\n{heat_ses}",
        )
        .replace("{observations_row}", observations_row)
        .replace("{group_mean_row}", group_mean_row)
    )


def main() -> None:
    wave5 = wave5_with_fuel_groups()
    specs = interleaved_specs(wave5, keep_kecamatans_on_both_sides(wave5))
    models = run_models(specs)

    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    (TABLE_OUTPUT / "table_g_fuel_shock_both_sides.tex").write_text(
        build_table(specs, models), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
