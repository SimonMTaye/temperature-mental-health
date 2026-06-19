from analysis.tables_v2.table_i_economics import ECONOMIC_OUTCOMES, winsorized_millions
from library.caching import run_regression_with_caching
from library.config import TABLE_OUTPUT
from library.specs import analysis_df, JOB_LOSS_MAIN, MAIN_TEMP_MEASURE, FE_WAVE
from library.dictionary import VARIABLE_LABELS
from library.render import RegressionSpec
from library.table_builder import coefficient_rows, make_row

balance_variables = [
    "age",
    "female",
    "edu_yrs",
    "married",
    "widowed",
    "cash_transfer_recipient",
    *ECONOMIC_OUTCOMES,
]

TABLE_TEMPLATE = r"""
\begin{tabular}{@{}lccccc}
\toprule
 & & \multicolumn{4}{c}{Regressor} \\
\cmidrule(lr){3-6}
 & Mean (SD) & Temperature & Job Loss &  Fuel Share &  \shortstack{Vehicle Owners\\Post Subsidy} \\ 
\cmidrule(lr){2-2} \cmidrule(lr){3-3} \cmidrule(lr){4-4} \cmidrule(lr){5-5} \cmidrule(lr){6-6} 
 & (1) & (2) & (3) & (4) & (5) \\
\midrule\addlinespace[2.5pt]
{coefficient_rows}
\midrule\addlinespace[2.5pt]
{observations_row}
\bottomrule
\end{tabular}
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
    balance_df = analysis_df.copy()
    for variable in ECONOMIC_OUTCOMES:
        balance_df[variable] = winsorized_millions(balance_df[variable])

    rows = {}
    observation_row = []
    for balance_variable in balance_variables:
        term_data = []
        term_data.append(
            (
                balance_df[balance_variable].mean(),
                balance_df[balance_variable].std(),
                0,
            ),
        )
        observation_row.append(len(balance_df))

        tests = [
            (MAIN_TEMP_MEASURE, balance_df),
            (JOB_LOSS_MAIN, balance_df),
            ("fuel_share_z_ifls4", balance_df),
            (
                "post_subsidy",
                balance_df[
                    balance_df["ifls5"] & (balance_df["urban_vehicle_hh_ifls4"] == 1)
                ].copy(),
            ),
        ]
        for term, data in tests:
            spec = RegressionSpec(
                title=f"{balance_variable} balance on {term}",
                formula=f"{balance_variable} ~ {term} | {FE_WAVE}",
                df=data,
                tags=frozenset(),
                show_terms=frozenset([term]),
            )
            result = run_regression_with_caching(
                spec,
            )
            term_data.append(
                (
                    float(result.coef_table.loc[term, "b"]),
                    float(result.coef_table.loc[term, "se"]),
                    float(result.coef_table.loc[term, "p"]),
                )
            )
            observation_row.append(int(result.stats["N"]))
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
