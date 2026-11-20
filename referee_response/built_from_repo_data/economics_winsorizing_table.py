"""Economic effects of the shocks with outcomes winsorized at the 95th or 99th percentile.

Run from the repo root:

    uv run python code/analysis/updates/economics_winsorizing_table.py

Responds to Reviewer 2, minor comment 1. Writes
output/tables/table_i_economics_winsorizing.tex.

The paper caps monetary outcomes at the 95th percentile (winsorized() in
table_i_economics.py). This table repeats the three economic regressions the
text cites with the cap moved to the 99th percentile:
    (1) Table I, palm shock:  household work income on palm x IFLS5.
    (2) Table I, job loss:    household work income on recent job loss.
    (3) Table C, post subsidy: transport and fuel spending on the post-cut
        indicator among IFLS5 urban vehicle-owning households.
As in the paper, the cap is computed on the full analysis sample before any
sample restriction. The lower tail is not capped: the 1st percentile of both
outcomes is zero.
"""

from analysis.tables_v2.table_i_economics import (
    earnings_variable,
    transportation_spending_variable,
)
from library.caching import run_regression_with_caching
from library.config import TABLE_OUTPUT
from library.render import RegressionSpec
from library.specs import CONTROLS, FE_NO_WAVE, FE_WAVE, JOB_LOSS_MAIN, analysis_df
from library.table_builder import coefficient_rows, make_row

CAPS = [(0.95, "95th Percentile (Paper)"), (0.99, "99th Percentile")]

TABLE_TEMPLATE = r"""
\begin{tabular}{@{}lccc}
\toprule
 & \multicolumn{2}{c}{Household Work Income} & \shortstack{Transport and\\Fuel Spending} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-4}
 & Palm Shock & Job Loss & \shortstack{Vehicle Owners\\Post Subsidy} \\
\cmidrule(lr){2-2} \cmidrule(lr){3-3} \cmidrule(lr){4-4}
 & (1) & (2) & (3) \\
\midrule\addlinespace[2.5pt]
{cap_panels}
\midrule\addlinespace[2.5pt]
Kecamatan & x & x & x \\
Month-Year & x & x & x \\
Demographic Controls & x & x & - \\
\midrule\addlinespace[2.5pt]
{observations_row}
\bottomrule
\end{tabular}
"""


def capped_specs(cap: float) -> list[tuple[RegressionSpec, str]]:
    """The three regressions with both outcomes capped at the given quantile."""
    df = analysis_df.copy()
    for outcome in [earnings_variable, transportation_spending_variable]:
        df[outcome] = df[outcome].clip(upper=df[outcome].quantile(cap))
    vehicle_owners = df[df["ifls5"] & (df["urban_vehicle_hh_ifls4"] == 1)].copy()
    regressions = [
        (
            f"{earnings_variable} ~ palm_farmer_hh_ifls4 * ifls5 + {CONTROLS} | {FE_NO_WAVE}",
            df,
            "palm_farmer_hh_ifls4:ifls5",
        ),
        (
            f"{earnings_variable} ~ {JOB_LOSS_MAIN} + {CONTROLS} | {FE_NO_WAVE}",
            df,
            JOB_LOSS_MAIN,
        ),
        (
            f"{transportation_spending_variable} ~ post_subsidy | {FE_WAVE}",
            vehicle_owners,
            "post_subsidy",
        ),
    ]
    return [
        (
            RegressionSpec(
                title=f"Economic effect, cap {cap}",
                formula=formula,
                df=data,
                tags=frozenset(["economics", "winsorizing"]),
                show_terms=frozenset([term]),
            ),
            term,
        )
        for formula, data, term in regressions
    ]


def build_table() -> str:
    panels = []
    for cap, label in CAPS:
        specs = capped_specs(cap)
        models = [run_regression_with_caching(spec) for spec, _ in specs]
        coefs, ses = coefficient_rows(
            "Shock Effect",
            [
                tuple(
                    float(model.coef_table.loc[term, key]) for key in ["b", "se", "p"]
                )
                for model, (_, term) in zip(models, specs)
            ],
            formatting=".2f",
        )
        panels += [
            rf"\multicolumn{{4}}{{l}}{{\text{{Winsorized at the {label}}}}} \\",
            r"\midrule",
            coefs,
            ses,
        ]
    return TABLE_TEMPLATE.replace("{cap_panels}", "\n".join(panels)).replace(
        "{observations_row}",
        make_row("Observations", [int(model.stats["N"]) for model in models]),
    )


def main() -> None:
    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    (TABLE_OUTPUT / "table_i_economics_winsorizing.tex").write_text(
        build_table(), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
