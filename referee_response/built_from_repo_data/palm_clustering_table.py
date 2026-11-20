"""Palm specifications under alternative clustering of the standard errors.

Run from the repo root:

    uv run python code/analysis/updates/palm_clustering_table.py

Responds to Reviewer 3, comment 10. Writes
output/tables/table_palm_clustering.tex.

For the three palm columns of Table B, the coefficient is unchanged and the
standard error is recomputed three ways:
    District                 kabupaten of interview (the paper's choice).
    District and Month-Year  two-way, adding the calendar month of interview.
    Baseline Province        the respondent's IFLS4 province, the level at
                             which palm exposure is defined (20 clusters).
Stars next to each standard error give the significance of the coefficient
under that clustering.

The models are fit with pyfixest directly rather than through
run_regression_with_caching, whose cache key does not include the clustering.
"""

import pyfixest as pf
from analysis.tables_v2.table_b_temperature_and_shock_effects import (
    TABLE_SPECS,
    wetbulb_specs,
)
from library.config import TABLE_OUTPUT
from library.table_builder import make_row, stars

BASELINE_PROVINCE = "baseline_province"
CLUSTERINGS = [
    ("District", "kabupaten_full_code"),
    ("District and Month-Year", "kabupaten_full_code+month_year"),
    ("Baseline Province", BASELINE_PROVINCE),
]
PALM_SPECS = TABLE_SPECS[:3]

TABLE_TEMPLATE = r"""
\begin{tabular}{@{}lccc}
\toprule
 & \multicolumn{3}{c}{CES-D z-score} \\
\cmidrule(lr){2-4}
 & Palm Shock & \shortstack{Palm Shock\\Panel} & \shortstack{Palm Shock\\Price Drop} \\
\cmidrule(lr){2-2} \cmidrule(lr){3-3} \cmidrule(lr){4-4}
 & (1) & (2) & (3) \\
\midrule\addlinespace[2.5pt]
{temperature_panel}
\midrule\addlinespace[2.5pt]
{wetbulb_panel}
\midrule\addlinespace[2.5pt]
Kecamatan & x & x & x \\
Month-Year & x & x & x \\
Individual & - & x & - \\
\midrule\addlinespace[2.5pt]
{observations_row}
\bottomrule
\end{tabular}
"""


def fit(spec_data: dict):
    df = spec_data["spec"].df.copy()
    baseline_province = df.loc[~df["ifls5"]].set_index("pidlink")["province_code"]
    df[BASELINE_PROVINCE] = df["pidlink"].map(baseline_province)
    df = df.dropna(subset=[BASELINE_PROVINCE])
    return pf.feols(
        spec_data["spec"].formula, data=df, vcov={"CRV1": "kabupaten_full_code"}
    )


def differential_term(model, spec_data: dict) -> str:
    wanted = {spec_data["group"], spec_data["post"], spec_data["heat"]}
    return next(term for term in model.coef().index if set(term.split(":")) == wanted)


def panel_rows(label: str, specs: list[dict], models) -> str:
    terms = [differential_term(model, spec) for model, spec in zip(models, specs)]
    rows = [
        rf"\multicolumn{{4}}{{l}}{{\text{{{label}}}}} \\",
        r"\midrule",
        make_row(
            "Differential Impact of Heat",
            [f"{model.coef()[term]:.3f}" for model, term in zip(models, terms)],
        ),
    ]
    for clustering_label, cluster in CLUSTERINGS:
        cells = []
        for model, term in zip(models, terms):
            model.vcov({"CRV1": cluster})
            cells.append(f"({model.se()[term]:.3f}){stars(model.pvalue()[term])}")
        rows.append(make_row(rf"\quad SE: {clustering_label}", cells))
    return "\n".join(rows)


def build_table() -> str:
    temperature_models = [fit(spec_data) for spec_data in PALM_SPECS]
    wetbulb = wetbulb_specs(PALM_SPECS)
    wetbulb_models = [fit(spec_data) for spec_data in wetbulb]
    return (
        TABLE_TEMPLATE.replace(
            "{temperature_panel}",
            panel_rows("7-day mean Temperature", PALM_SPECS, temperature_models),
        )
        .replace(
            "{wetbulb_panel}",
            panel_rows("7-day mean Wet-bulb", wetbulb, wetbulb_models),
        )
        .replace(
            "{observations_row}",
            make_row("Observations", [model._N for model in temperature_models]),
        )
    )


def main() -> None:
    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    (TABLE_OUTPUT / "table_palm_clustering.tex").write_text(
        build_table(), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
