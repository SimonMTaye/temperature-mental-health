from library.config import TABLE_OUTPUT
from library.specs import (
    jobloss,
    JOB_LOSS_MAIN,
    MAIN_TEMP_MEASURE,
    analysis_df,
    CONTROLS,
    FE_NO_WAVE,
    RegressionSpec,
)
from library.render import make_regression_table, render_table_to_latex


def make_table() -> None:
    analysis_df["jobloss_occupation_sector"] = analysis_df[
        "current_job_sector"
    ].combine_first(analysis_df["job_loss_sector"])
    occupation_sector_codes = sorted(analysis_df["jobloss_occupation_sector"].unique())
    occupation_sector_terms = []
    for sector_code in occupation_sector_codes[1:]:
        sector_column = f"jobloss_occ_sector_{sector_code}"
        analysis_df[sector_column] = (
            analysis_df["jobloss_occupation_sector"].eq(sector_code).astype("Int32")
        )
    occupation_sector_terms.append(sector_column)

    occupation_sector_slope_terms = " + ".join(
        f"{sector_column} + {sector_column}:{MAIN_TEMP_MEASURE}"
        for sector_column in occupation_sector_terms
    )
    jobloss_slope_specs = [
        jobloss,
        RegressionSpec(
            title="+ Sector x Heat",
            formula=(
                f"cesd_z ~ {JOB_LOSS_MAIN} * {MAIN_TEMP_MEASURE} "
                f"+ {occupation_sector_slope_terms} + {CONTROLS} | {FE_NO_WAVE}"
            ),
            df=analysis_df,
            tags=frozenset(["job-loss", "slope-robustness", "occupation-sector"]),
            show_terms=jobloss.show_terms,
        ),
        RegressionSpec(
            title="+ Age x Heat",
            formula=(
                f"cesd_z ~ {JOB_LOSS_MAIN} * {MAIN_TEMP_MEASURE} "
                f"+ age:{MAIN_TEMP_MEASURE} + {CONTROLS} | {FE_NO_WAVE}"
            ),
            df=analysis_df,
            tags=frozenset(["job-loss", "slope-robustness", "age-slope"]),
            show_terms=jobloss.show_terms,
        ),
        RegressionSpec(
            title="+ Education x Heat",
            formula=(
                f"cesd_z ~ {JOB_LOSS_MAIN} * {MAIN_TEMP_MEASURE} "
                f"+ edu_yrs:{MAIN_TEMP_MEASURE} + {CONTROLS} | {FE_NO_WAVE}"
            ),
            df=analysis_df,
            tags=frozenset(["job-loss", "slope-robustness", "education-slope"]),
            show_terms=jobloss.show_terms,
        ),
    ]
    rename_dict = {
        JOB_LOSS_MAIN: "Job Loss",
        MAIN_TEMP_MEASURE: "Heat",
    }
    table = make_regression_table(
        jobloss_slope_specs,
        rename=rename_dict,
        keep=["Job Loss:Heat"],
        titles=[spec.title for spec in jobloss_slope_specs],
    )
    render_table_to_latex(
        table,
        TABLE_OUTPUT / "table_f_job_loss_controls.tex",
    )


if __name__ == "__main__":
    make_table()
