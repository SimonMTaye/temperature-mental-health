from library.config import TABLE_OUTPUT
from library.specs import (
    update_formula_search_replace,
    jobloss,
    JOB_LOSS_MAIN,
    MAIN_TEMP_MEASURE,
    analysis_df,
)
from library.render import make_regression_table, render_table_to_latex
from dataclasses import replace

JOB_LOSS_WINDOWS = [
    ("3 months", "job_loss_90d"),
    ("6 months", "job_loss_180d"),
    ("9 months", "job_loss_270d"),
    ("12 months", "job_loss_365d"),
    ("3 years", "job_loss_1095d"),
    ("5 yr", "job_loss_1825d"),
]


def make_table() -> None:
    jobloss_robustness_specs = [
        update_formula_search_replace(jobloss, JOB_LOSS_MAIN, job_loss_var)
        for _, job_loss_var in JOB_LOSS_WINDOWS
    ]
    jobloss_complete_case_columns = [
        "pidlink",
        "cesd_z",
        MAIN_TEMP_MEASURE,
        "age",
        "female",
        "edu_yrs",
        "married",
        "widowed",
        "month",
        "year",
        "ifls5",
        "gadm_fullcode",
        # "palm_farmer_hh_ifls4",
    ]
    job_loss_df = analysis_df.dropna(
        subset=[*jobloss_complete_case_columns, JOB_LOSS_MAIN]
    )
    jobloss_robustness_specs = [
        replace(spec, df=job_loss_df) for spec in jobloss_robustness_specs
    ]

    jobloss_treated_people = [
        analysis_df[job_loss_var].eq(1).sum() for _, job_loss_var in JOB_LOSS_WINDOWS
    ]
    rename_dict = {job_loss_var: "Job Loss" for _, job_loss_var in JOB_LOSS_WINDOWS}
    # Append
    rename_dict[MAIN_TEMP_MEASURE] = "Heat"
    table = make_regression_table(
        jobloss_robustness_specs,
        rename=rename_dict,
        keep=["Job Loss:Heat"],
        titles=[f"{label}" for label, _ in JOB_LOSS_WINDOWS],
        custom_model_stats={"People with job loss": jobloss_treated_people},
    )
    render_table_to_latex(
        table,
        TABLE_OUTPUT / "table_e_job_loss_window_robustness.tex",
    )


if __name__ == "__main__":
    make_table()
