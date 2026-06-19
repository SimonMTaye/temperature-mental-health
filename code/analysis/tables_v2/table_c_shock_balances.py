import pyfixest as pf

from library.config import TABLE_OUTPUT
from library.specs import analysis_df, JOB_LOSS_MAIN, MAIN_TEMP_MEASURE, FE_WAVE
from library.dictionary import VARIABLE_LABELS
from library.table_builder import coefficient_rows, make_row

balance_variables_demographic = [
    "age",
    "female",
    "edu_yrs",
    "married",
    "widowed",
]

TABLE_TEMPLATE = r"""\resizebox{\linewidth}{!}{
\begin{tabular}{@{}lccccc}
\toprule
 & & \multicolumn{4}{c}{Regressor} \\
\cmidrule(lr){3-6}
 & Mean (SD) & Temperature & Job Loss &  Fuel Share &  \shortstack{Vehicle Owners\\Post Subsidy} \\ 
\midrule\addlinespace[2.5pt]
{coefficient_rows}
\midrule\addlinespace[2.5pt]
{observations_row}
\bottomrule
\end{tabular}
}
"""

# Goal is a balance table
# One row per balance demographic variable
# Columns are
#   Overall Mean (sd below it)
#   Temperature -> MAIN_TEMPERATURE_MEASURE
#   Job Loss -> JOB_LOSS_MAIN
#   Palm Farmers -> palm_farmer_hh_ifls4
#   Fuel Share -> fuel_share_ifls4
#   Urban Vehicle Owners (coefficient, se, p-value) -> urban_vehicle_hh_ifls4
#   Post Subsidy -> post_subsidy (data should be limited to wave 5)


def make_table() -> None:
    rows = {}
    observation_row = []
    for balance_variable in balance_variables_demographic:
        term_data = []
        term_data.append(
            (
                analysis_df[balance_variable].mean(),
                analysis_df[balance_variable].std(),
                0,
            ),
        )
        observation_row.append(len(analysis_df))

        tests = [
            (MAIN_TEMP_MEASURE, analysis_df),
            (JOB_LOSS_MAIN, analysis_df),
            ("fuel_share_ifls4", analysis_df),
            (
                "post_subsidy",
                analysis_df[
                    analysis_df["ifls5"] & (analysis_df["urban_vehicle_hh_ifls4"] == 1)
                ].copy(),
            ),
        ]
        for term, data in tests:
            result = pf.feols(
                f"{balance_variable} ~ {term} | {FE_WAVE}",
                data=data,
                vcov={"CRV1": "gadm_fullcode"},
            )
            term_data.append(
                (
                    float(result.coef().loc[term]),
                    float(result.se().loc[term]),
                    float(result.pvalue().loc[term]),
                )
            )
            observation_row.append(int(result._N))
        rows[VARIABLE_LABELS[balance_variable]] = term_data

    coef_rows = []
    for item, stats in rows.items():
        coef_rows.extend(coefficient_rows(item, stats))
    table = TABLE_TEMPLATE.replace("{coefficient_rows}", "\n".join(coef_rows))
    table = table.replace(
        "{observations_row}",
        make_row("Observations", observation_row[: len(tests) + 1]),
    )
    with open(TABLE_OUTPUT / "table_c_shock_balances.tex", "w") as f:
        f.write(table)


if __name__ == "__main__":
    make_table()
