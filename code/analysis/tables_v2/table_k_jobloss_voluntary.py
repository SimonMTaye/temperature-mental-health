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
        job_loss_involuntary=lambda df: (
            df[JOB_LOSS_MAIN] * df["involuntary_loss_5y"]
        ).astype("Int32"),
        job_loss_voluntary=lambda df: 1 - df.job_loss_involuntary,
    )
    rename_dict = {
        JOB_LOSS_MAIN: "Job Loss",
        "job_loss_involuntary": "Job Loss",
        "job_loss_voluntary": "Job Loss",
        MAIN_TEMP_MEASURE: "Heat",
    }
    jobloss_specs = [
        jobloss,
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
    ]
    table = make_regression_table(
        jobloss_specs,
        rename=rename_dict,
        keep=["Job Loss:Heat"],
        titles=[
            "Job Loss",
            r"\shortstack{Job Loss\\Involuntary}",
            r"\shortstack{Job Loss\\Voluntary}",
        ],
    )
    render_table_to_latex(
        table,
        TABLE_OUTPUT / "table_k_jobloss_voluntary.tex",
    )


if __name__ == "__main__":
    make_table()
