"""Palm panel specification with palm-specific seasonality and province heat slopes.

Run from the repo root:

    uv run python code/analysis/updates/palm_identification_controls.py

Responds to Reviewer 3, comment 1. Writes one table per palm exposure measure:
output/tables/table_palm_identification_proxy.tex (Table B column 2, palm
proxy) and output/tables/table_palm_identification_price_gap.tex (same
individual-FE specification with palm_price_gap_z as the exposure).

Columns:
    (1) Baseline   the individual-FE palm specification, unchanged.
    (2) + palm proxy x calendar-month fixed effects, so palm households can
        have their own seasonal pattern in CES-D.
    (3) + baseline-province-specific heat slopes, so the heat effect can
        differ by the province where the respondent lived in IFLS4.
    (4) Both.

Province-by-wave fixed effects are deliberately not included: palm exposure
is defined at the province level, so they would absorb the shock itself.
Identification is otherwise unchanged: kecamatan, month-by-year and
individual fixed effects, district-clustered standard errors.
"""

from dataclasses import replace

from analysis.tables_v2.table_b_temperature_and_shock_effects import (
    coefficient_stats,
    regression_runner,
    wetbulb_specs,
)
from library.config import TABLE_OUTPUT
from library.specs import (
    MAIN_TEMP_MEASURE,
    analysis_df,
    palm_shock_panel,
    update_formula_search_replace,
)
from library.table_builder import coefficient_rows, make_row

PALM_PROXY = "palm_farmer_hh_ifls4"
PALM_PRICE_GAP = "palm_price_gap_z"
PALM_MONTH_FE = "palm_month"
BASELINE_PROVINCE = "baseline_province"

# (palm x month FE, baseline province x heat)
COLUMNS = [(False, False), (True, False), (False, True), (True, True)]

TABLE_TEMPLATE = r"""
\begin{tabular}{@{}lcccc}
\toprule
 & \multicolumn{4}{c}{CES-D z-score} \\
\cmidrule(lr){2-5}
 & Baseline & \shortstack{Palm x\\Month} & \shortstack{Province x\\Heat} & Both \\
\cmidrule(lr){2-2} \cmidrule(lr){3-3} \cmidrule(lr){4-4} \cmidrule(lr){5-5}
 & (1) & (2) & (3) & (4) \\
\midrule\addlinespace[2.5pt]
{temperature_panel}
\midrule\addlinespace[2.5pt]
{wetbulb_panel}
\midrule\addlinespace[2.5pt]
Palm x Calendar Month & - & x & - & x \\
Baseline Province x Heat & - & - & x & x \\
Kecamatan & x & x & x & x \\
Month-Year & x & x & x & x \\
Individual & x & x & x & x \\
\midrule\addlinespace[2.5pt]
{observations_row}
\bottomrule
\end{tabular}
"""


def panel_df():
    """Add the baseline province and the palm x calendar-month cell."""
    df = analysis_df.copy()
    baseline_province = df.loc[~df["ifls5"]].set_index("pidlink")["province_code"]
    df[BASELINE_PROVINCE] = df["pidlink"].map(baseline_province)
    df[PALM_MONTH_FE] = (
        df[PALM_PROXY].astype("Int64").astype("string")
        + "_"
        + df["month"].astype("Int64").astype("string")
    )
    # The price-gap sample is a superset; keep the palm-proxy panel sample for both tables.
    return df.dropna(subset=[PALM_PROXY, BASELINE_PROVINCE])


def column_specs(group: str) -> list[dict]:
    df = panel_df()
    base = replace(
        update_formula_search_replace(palm_shock_panel, PALM_PROXY, group), df=df
    )
    specs = []
    for palm_month, province_heat in COLUMNS:
        rhs, fixed_effects = base.formula.split("|")
        if province_heat:
            rhs += f" + C({BASELINE_PROVINCE}):{MAIN_TEMP_MEASURE} "
        if palm_month:
            fixed_effects += f" + {PALM_MONTH_FE}"
        spec = replace(base, formula=f"{rhs}|{fixed_effects}")
        specs.append(
            {"spec": spec, "group": group, "post": "ifls5", "heat": MAIN_TEMP_MEASURE}
        )
    return specs


def panel_rows(label: str, models) -> str:
    coefs, ses = coefficient_rows(
        "Differential Impact of Heat",
        [coefficient_stats(model, "group:post:heat") for model in models],
    )
    return "\n".join(
        [rf"\multicolumn{{5}}{{l}}{{\text{{{label}}}}} \\", r"\midrule", coefs, ses]
    )


def build_table(group: str) -> str:
    specs = column_specs(group)
    temperature_models = regression_runner(specs)
    wetbulb_models = regression_runner(wetbulb_specs(specs))
    observations_row = make_row(
        "Observations", [f"{int(model.stats['N']):,}" for model in temperature_models]
    )
    return (
        TABLE_TEMPLATE.replace(
            "{temperature_panel}",
            panel_rows("7-day mean Temperature", temperature_models),
        )
        .replace("{wetbulb_panel}", panel_rows("7-day mean Wet-bulb", wetbulb_models))
        .replace("{observations_row}", observations_row)
    )


def main() -> None:
    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, group in [("proxy", PALM_PROXY), ("price_gap", PALM_PRICE_GAP)]:
        (TABLE_OUTPUT / f"table_palm_identification_{name}.tex").write_text(
            build_table(group), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
