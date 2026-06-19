from library.render import RegressionSpec
from library.specs import update_formula_search_replace
from library.config import TABLE_OUTPUT
from copy import deepcopy
from analysis.tables_v2.table_b_temperature_and_shock_effects import (
    TABLE_SPECS,
    panel_rows,
    regression_runner,
    make_row,
)

TABLE_TEMPLATE = r"""
\begin{tabular}{@{}lcccccc}
\toprule
 & \multicolumn{6}{c}{CES-D z-score} \\
\cmidrule(lr){2-7}
 & Palm Shock & \shortstack{Palm Shock\\Panel} & \shortstack{Palm Shock\\Price Drop} & \shortstack{Fuel Cut\\Urban Vehicle Owners} & \shortstack{Fuel Cut\\Fuel Share} &  Job Loss \\
\cmidrule(lr){2-2} \cmidrule(lr){3-3} \cmidrule(lr){4-4} \cmidrule(lr){5-5} \cmidrule(lr){6-6} \cmidrule(lr){7-7}
 & (1) & (2) & (3) & (4) & (5) & (6) \\
\midrule\addlinespace[2.5pt]
{somatic}
\midrule\addlinespace[2.5pt]
{depressed}
\midrule\addlinespace[2.5pt]
{positive}
\midrule\addlinespace[2.5pt]
Kecamatan & x & x & x & x & x & x \\
Wave & x & x & x & - & - & x \\
Month & x & x & x & x & x & x \\
Individual & - & x & - & - & - & - \\
Year  & x & x & x & x & x & x \\
\midrule\addlinespace[2.5pt]
{observations_row}
\bottomrule
\end{tabular}
\begin{minipage}{\linewidth}
Significance levels: * p < 0.1, ** p < 0.05, *** p < 0.01. Format of coefficient cell: Coefficient   (Std. Error)\\
\end{minipage}
"""


# "somatic_z",
# "depraffect_z"
# "posaffect_z"
def build_table() -> str:
    depression_specs = deepcopy(TABLE_SPECS)
    for spec in depression_specs:
        assert isinstance(spec["spec"], RegressionSpec)
        spec["spec"] = update_formula_search_replace(
            spec["spec"], "cesd_z", "depraffect_z"
        )
    somatic_specs = deepcopy(TABLE_SPECS)
    for spec in somatic_specs:
        assert isinstance(spec["spec"], RegressionSpec)
        spec["spec"] = update_formula_search_replace(
            spec["spec"], "cesd_z", "somatic_z"
        )
    positive_specs = deepcopy(TABLE_SPECS)
    for spec in positive_specs:
        assert isinstance(spec["spec"], RegressionSpec)
        spec["spec"] = update_formula_search_replace(
            spec["spec"], "cesd_z", "posaffect_z"
        )
    depression_models = regression_runner(depression_specs)
    somatic_models = regression_runner(somatic_specs)
    positive_models = regression_runner(positive_specs)

    observations_rows = make_row(
        "Observations", [f"{int(model.stats['N']):,}" for model in depression_models]
    )

    return (
        TABLE_TEMPLATE.replace(
            "{depressed}",
            panel_rows("Depressed-affect CES-D factor", depression_models),
        )
        .replace(
            "{somatic}",
            panel_rows("Somatic / activity-related CES-D factor", somatic_models),
        )
        .replace(
            "{positive}", panel_rows("Positive-affect CES-D factor", positive_models)
        )
        .replace("{observations_row}", observations_rows)
    )


TABLE_FILE = TABLE_OUTPUT / "table_l_temperature_shock_cesd_breakdown.tex"


def make_table() -> None:
    TABLE_FILE.write_text(build_table())


if __name__ == "__main__":
    make_table()
