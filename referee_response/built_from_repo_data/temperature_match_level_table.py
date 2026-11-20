"""Re-run Table B without respondents whose temperature is assigned at a coarser level.

Run from the repo root:

    uv run python code/analysis/updates/temperature_match_level_table.py

Responds to Reviewer 3, comment 9. Writes
output/tables/table_b_kecamatan_match_only.tex.

About 1% of respondents could be matched only to a kabupaten or province
polygon (match_level != "kecamatan"), so their temperature is measured at
that coarser level. Each Table B column is estimated twice: on the full
sample (reproducing Table B) and after dropping those respondents. In the
individual-FE column, a respondent who loses one wave drops out entirely.
Specifications are otherwise those of Table B.
"""

from dataclasses import replace

from analysis.tables_v2.table_b_temperature_and_shock_effects import (
    TABLE_SPECS,
    differential_impact_row,
    regression_runner,
    wetbulb_specs,
)
from library.config import TABLE_OUTPUT
from library.table_builder import make_row

TABLE_TEMPLATE = r"""
\begin{tabular}{@{}lccccccc}
\toprule
 & \multicolumn{7}{c}{CES-D z-score} \\
\cmidrule(lr){2-8}
 & Palm Shock & \shortstack{Palm Shock\\Panel} & \shortstack{Palm Shock\\Price Drop} & Urban Vehicle Owners & \shortstack{Urban Vehicle Owners\\No Cash Transfer} & \shortstack{Urban Vehicle Owners\\Cash Transfer} & Job Loss \\
\cmidrule(lr){2-2} \cmidrule(lr){3-3} \cmidrule(lr){4-4} \cmidrule(lr){5-5} \cmidrule(lr){6-6} \cmidrule(lr){7-7} \cmidrule(lr){8-8}
 & (1) & (2) & (3) & (4) & (5) & (6) & (7) \\
\midrule\addlinespace[2.5pt]
{temperature_panel}
\midrule\addlinespace[2.5pt]
{wetbulb_panel}
\midrule\addlinespace[2.5pt]
{observations_rows}
\bottomrule
\end{tabular}
"""


def kecamatan_match_specs(specs: list[dict]) -> list[dict]:
    """Keep only respondents whose temperature is measured at the kecamatan level."""
    restricted = []
    for spec_data in specs:
        df = spec_data["spec"].df
        df = df.loc[df["match_level"].eq("kecamatan")]
        restricted.append({**spec_data, "spec": replace(spec_data["spec"], df=df)})
    return restricted


def panel_rows(label: str, full_models, restricted_models) -> str:
    rows = [rf"\multicolumn{{8}}{{l}}{{\text{{{label}}}}} \\", r"\midrule"]
    for sample, models in [
        ("All Observations", full_models),
        ("Kecamatan-Level Temperature Only", restricted_models),
    ]:
        coefs, ses = differential_impact_row(models)
        rows += [coefs.replace("Differential Impact of Heat", sample, 1), ses]
    return "\n".join(rows)


def observations_row(label: str, models) -> str:
    return make_row(label, [f"{int(model.stats['N']):,}" for model in models])


def build_table() -> str:
    restricted_specs = kecamatan_match_specs(TABLE_SPECS)
    full_models = regression_runner(TABLE_SPECS)
    restricted_models = regression_runner(restricted_specs)
    return (
        TABLE_TEMPLATE.replace(
            "{temperature_panel}",
            panel_rows("7-day mean Temperature", full_models, restricted_models),
        )
        .replace(
            "{wetbulb_panel}",
            panel_rows(
                "7-day mean Wet-bulb",
                regression_runner(wetbulb_specs()),
                regression_runner(wetbulb_specs(restricted_specs)),
            ),
        )
        .replace(
            "{observations_rows}",
            observations_row("Observations, All", full_models)
            + "\n"
            + observations_row("Observations, Kecamatan-Level Only", restricted_models),
        )
    )


def main() -> None:
    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    (TABLE_OUTPUT / "table_b_kecamatan_match_only.tex").write_text(
        build_table(), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
