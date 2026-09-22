from __future__ import annotations

from dataclasses import replace
import re

import numpy as np
import pandas as pd

from library.caching import run_regression_with_caching
from library.config import TABLE_OUTPUT
from library.specs import (
    JOB_LOSS_MAIN,
    MAIN_TEMP_MEASURE,
    fuel_shock_urban_vehicle,
    wave5_df,
    jobloss,
    palm_shock,
    update_formula_search_replace,
)
from library.table_builder import make_row, coefficient_rows

TABLE_FILE = TABLE_OUTPUT / "table_n_interaction_effects_sleep.tex"
KABUPATEN_CLUSTER = "kabupaten_full_code"
WETBULB_MEASURE = "wetbulb_7d"
FORMULA_SPLIT_PATTERN = re.compile(r"(\s+|[~+*:/|()])")

PALM_PRICE_SHOCK = update_formula_search_replace(
    palm_shock,
    "palm_farmer_hh_ifls4",
    "palm_price_gap_z",
)

urban_nocash = replace(
    update_formula_search_replace(
        fuel_shock_urban_vehicle,
        "urban_vehicle_hh_ifls4",
        "urban_vehicle_transfer_nonrecipient_ifls4",
    ),
    df=wave5_df,
)

urban_cash = replace(
    update_formula_search_replace(
        fuel_shock_urban_vehicle,
        "urban_vehicle_hh_ifls4",
        "urban_vehicle_transfer_recipient_ifls4",
    ),
    df=wave5_df,
)

SLEEP_TABLE_SPECS = [
    {
        "spec": fuel_shock_urban_vehicle,
        "group": "urban_vehicle_hh_ifls4",
        "post": "post_subsidy",
        "heat": MAIN_TEMP_MEASURE,
        "label": r"Urban Vehicle Owners",
    },
    {
        "spec": jobloss,
        "group": JOB_LOSS_MAIN,
        "post": None,
        "heat": MAIN_TEMP_MEASURE,
        "label": r"Job Loss",
    },
]

# \resizebox{\linewidth}{!}{%
TABLE_TEMPLATE = r"""
\begin{tabular}{@{}lcc}
\toprule
 & \multicolumn{2}{c}{Sleep Hours} \\
\cmidrule(lr){2-3}
& Urban Vehicle Owners & Job Loss \\
\cmidrule(lr){2-2} \cmidrule(lr){3-3} 
 & (1) & (2) \\
{temperature_panel}
\midrule\addlinespace[2.5pt]
Kecamatan & x & x \\
Month-Year & x & x \\
\midrule\addlinespace[2.5pt]
{observations_row}
{group_mean_row}
\bottomrule
\end{tabular}
"""


def replace_terms(text: str, replacements: dict[str, str]) -> str:
    parts = FORMULA_SPLIT_PATTERN.split(text)
    for index, part in enumerate(parts):
        if part in replacements:
            parts[index] = replacements[part]
    return "".join(parts)


def standardize_model_terms(model, replacements: dict[str, str]):
    coef_table = model.coef_table.copy()
    coef_table.index = pd.Index(
        [replace_terms(term, replacements) for term in coef_table.index.astype(str)],
        name=coef_table.index.name,
    )

    vcov = model.vcov.copy()
    vcov.index = pd.Index(
        [replace_terms(term, replacements) for term in vcov.index.astype(str)],
        name=vcov.index.name,
    )
    vcov.columns = pd.Index(
        [replace_terms(term, replacements) for term in vcov.columns.astype(str)],
        name=vcov.columns.name,
    )
    return replace(model, coef_table=coef_table, vcov=vcov)


def regression_runner(specs: list[dict]):
    models = []
    for spec_data in specs:
        model = run_regression_with_caching(
            spec_data["spec"],
            vcov_type=KABUPATEN_CLUSTER,
        )
        replacements = {
            spec_data["group"]: "group",
            spec_data["heat"]: "heat",
        }
        if spec_data["post"] is not None:
            replacements[spec_data["post"]] = "post"
        models.append(standardize_model_terms(model, replacements))
    return models


def coefficient_stats(model, term: str) -> tuple[float, float, float]:
    if term not in model.coef_table.index:
        return np.nan, np.nan, np.nan
    coefficient = float(model.coef_table.loc[term, "b"])
    standard_error = float(model.coef_table.loc[term, "se"])
    p_value = float(model.coef_table.loc[term, "p"])
    return coefficient, standard_error, p_value


def differential_impact_row(models) -> tuple[str, str]:
    stats = []
    for model in models:
        term = (
            "group:post:heat"
            if "group:post:heat" in model.coef_table.index
            else "group:heat"
        )
        stats.append(coefficient_stats(model, term))
    return coefficient_rows("Differential Impact of Heat", stats)


def panel_rows(models) -> str:
    differential_coefs, differential_ses = differential_impact_row(models)
    heat_coefs, heat_ses = coefficient_rows(
        "Heat", [coefficient_stats(model, "heat") for model in models]
    )
    return "\n".join(
        [
            r"\midrule",
            differential_coefs,
            differential_ses,
            heat_coefs,
            heat_ses,
        ]
    )


def stressed_group_proportion_row(specs: list[dict]) -> str:
    return make_row(
        "Stressed Group Proportion",
        [
            f"{spec_data['spec'].df[spec_data['group']].mean():.3f}"
            for spec_data in specs
        ],
    )


def build_table() -> str:
    sleep_specs = [
        {
            **spec_data,
            "spec": update_formula_search_replace(
                spec_data["spec"],  # ty:ignore[invalid-argument-type]
                "cesd_z",
                "sleep_dur_h",
            ),
        }
        for spec_data in SLEEP_TABLE_SPECS
    ]
    sleep_models = regression_runner(sleep_specs)
    observations_rows = make_row(
        "Observations", [f"{int(model.stats['N']):,}" for model in sleep_models]
    )
    return (
        TABLE_TEMPLATE.replace(
            "{temperature_panel}",
            panel_rows(sleep_models),
        )
        .replace("{observations_row}", observations_rows)
        .replace("{group_mean_row}", stressed_group_proportion_row(sleep_specs))
    )


def make_table() -> None:
    TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TABLE_FILE.write_text(build_table(), encoding="utf-8")


if __name__ == "__main__":
    make_table()
