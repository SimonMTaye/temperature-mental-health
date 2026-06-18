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
    fuel_shock_fuel_share,
    fuel_shock_urban_vehicle,
    jobloss,
    palm_shock,
    palm_shock_panel,
    update_formula_search_replace,
)
from library.table_builder import make_row, coefficient_rows

TABLE_FILE = TABLE_OUTPUT / "table_b_temperature_and_shock_effects.tex"
KABUPATEN_CLUSTER = "kabupaten_full_code"
WETBULB_MEASURE = "wetbulb_7d"
FORMULA_SPLIT_PATTERN = re.compile(r"(\s+|[~+*:/|()])")

PALM_PRICE_SHOCK = update_formula_search_replace(
    palm_shock,
    "palm_farmer_hh_ifls4",
    "palm_price_gap_z",
)

TABLE_SPECS = [
    {
        "spec": palm_shock,
        "group": "palm_farmer_hh_ifls4",
        "post": "ifls5",
        "heat": MAIN_TEMP_MEASURE,
    },
    {
        "spec": palm_shock_panel,
        "group": "palm_farmer_hh_ifls4",
        "post": "ifls5",
        "heat": MAIN_TEMP_MEASURE,
    },
    {
        "spec": PALM_PRICE_SHOCK,
        "group": "palm_price_gap_z",
        "post": "ifls5",
        "heat": MAIN_TEMP_MEASURE,
    },
    {
        "spec": fuel_shock_fuel_share,
        "group": "fuel_share_ifls4",
        "post": "post_subsidy",
        "heat": MAIN_TEMP_MEASURE,
    },
    {
        "spec": fuel_shock_urban_vehicle,
        "group": "urban_vehicle_hh_ifls4",
        "post": "post_subsidy",
        "heat": MAIN_TEMP_MEASURE,
    },
    {
        "spec": jobloss,
        "group": JOB_LOSS_MAIN,
        "post": None,
        "heat": MAIN_TEMP_MEASURE,
    },
]

TABLE_TEMPLATE = r"""\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lcccccc}
\toprule
 & \multicolumn{6}{c}{CES-D z-score} \\
\cmidrule(lr){2-7}
 & Palm Shock & \shortstack{Palm Shocks\\Panel} & \shortstack{Palm Shock\\Price Drop} & \shortstack{Fuel Cut\\Fuel Share} & \shortstack{Fuel Cut\\Urban Vehicle Owners} & Job Loss \\
 & (1) & (2) & (3) & (4) & (5) & (6) \\
\midrule\addlinespace[2.5pt]
{temperature_panel}
\midrule\addlinespace[2.5pt]
{wetbulb_panel}
\midrule\addlinespace[2.5pt]
Kecamatan & x & x & x & x & x & x \\
Wave & x & x & x & - & - & x \\
Month & x & x & x & x & x & x \\
Individual & - & x & - & - & - & - \\
Year  & x & x & x & x & x & x \\
\midrule\addlinespace[2.5pt]
{observations_row}
{r2_row}
\bottomrule
\end{tabular}%
}
\begin{minipage}{\linewidth}
Significance levels: * p < 0.1, ** p < 0.05, *** p < 0.01. Format of coefficient cell: Coefficient   (Std. Error)\\
\end{minipage}
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


def wetbulb_specs() -> list[dict]:
    specs = []
    for spec_data in TABLE_SPECS:
        specs.append(
            {
                **spec_data,
                "spec": update_formula_search_replace(
                    spec_data["spec"],
                    MAIN_TEMP_MEASURE,
                    WETBULB_MEASURE,
                ),
                "heat": WETBULB_MEASURE,
            }
        )
    return specs


def panel_rows(label: str, models) -> str:
    differential_coefs, differential_ses = differential_impact_row(models)
    heat_coefs, heat_ses = coefficient_rows(
        "Heat", [coefficient_stats(model, "heat") for model in models]
    )
    return "\n".join(
        [
            rf"\multicolumn{{7}}{{l}}{{\text{{{label}}}}} \\",
            r"\midrule",
            differential_coefs,
            differential_ses,
            heat_coefs,
            heat_ses,
        ]
    )


def build_table() -> str:
    temperature_models = regression_runner(TABLE_SPECS)
    wetbulb_models = regression_runner(wetbulb_specs())
    observations_rows = make_row(
        "Observations", [f"{int(model.stats['N']):,}" for model in temperature_models]
    )
    r2_rows = make_row(
        "R²", [f"{model.stats['r2']:.3f}" for model in temperature_models]
    )

    return (
        TABLE_TEMPLATE.replace(
            "{temperature_panel}",
            panel_rows("7-day mean Temperature", temperature_models),
        )
        .replace("{wetbulb_panel}", panel_rows("7-day mean Wet Bulb", wetbulb_models))
        .replace("{observations_row}", observations_rows)
        .replace("{r2_row}", r2_rows)
    )


if __name__ == "__main__":
    TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TABLE_FILE.write_text(build_table(), encoding="utf-8")
