from library.config import TABLE_OUTPUT
from library.specs import (
    jobloss,
    JOB_LOSS_MAIN,
    MAIN_TEMP_MEASURE,
    analysis_df,
    update_formula_search_replace,
)
from library.render import make_regression_table, render_table_to_latex
from dataclasses import replace


def make_table() -> None:
    jobloss_df = analysis_df.copy().assign(
        kabupaten_full_code=lambda df: df["gadm_fullcode"].astype("int64") // 1000,
        job_loss_involuntary=lambda df: (
            df[JOB_LOSS_MAIN] * df["job_loss_involuntary"]
        ).astype("Int32"),
        job_loss_voluntary=lambda df: (
            df[JOB_LOSS_MAIN] * (df["job_loss_family"])
        ).astype("Int32"),
        job_loss_sick=lambda df: (
            df[JOB_LOSS_MAIN]
            * (1 - df["job_loss_family"])
            * (1 - df["job_loss_involuntary"])
        ).astype("Int32"),
    )
    rename_dict = {
        JOB_LOSS_MAIN: "Job Loss",
        "job_loss_involuntary": "Job Loss",
        "job_loss_voluntary": "Job Loss",
        "job_loss_sick": "Job Loss",
        MAIN_TEMP_MEASURE: "Heat",
    }
    jobloss_specs = [
        replace(jobloss, df=jobloss_df),
        replace(
            update_formula_search_replace(
                jobloss, JOB_LOSS_MAIN, "job_loss_involuntary"
            ),
            df=jobloss_df,
        ),
        replace(
            update_formula_search_replace(jobloss, JOB_LOSS_MAIN, "job_loss_voluntary"),
            df=jobloss_df,
        ),
        replace(
            update_formula_search_replace(jobloss, JOB_LOSS_MAIN, "job_loss_sick"),
            df=jobloss_df,
        ),
    ]
    jobloss_treated_people = [
        jobloss_df[job_loss_var].eq(1).sum()
        for job_loss_var in [
            JOB_LOSS_MAIN,
            "job_loss_involuntary",
            "job_loss_voluntary",
            "job_loss_sick",
        ]
    ]
    table = make_regression_table(
        jobloss_specs,
        rename=rename_dict,
        keep=["Job Loss:Heat"],
        titles=[
            "Job Loss",
            r"\shortstack{Job Loss\\Involuntary}",
            r"\shortstack{Job Loss\\Family}",
            r"\shortstack{Job Loss\\Sickness}",
        ],
        custom_model_stats={"People with job loss": jobloss_treated_people},
    )
    render_table_to_latex(
        table,
        TABLE_OUTPUT / "table_k_jobloss_voluntary.tex",
    )


if __name__ == "__main__":
    make_table()
