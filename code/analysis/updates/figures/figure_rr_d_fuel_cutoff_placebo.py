"""Plot fuel-shock coefficients for placebo cutoffs in one- and two-week steps around November 18, 2014.

Run from the repo root:

    uv run python code/analysis/updates/figures/figure_rr_d_fuel_cutoff_placebo.py

Writes two figures (PDF and PNG) to output/figures/:
    figure_rr_d_fuel_cutoff_placebo            cutoffs every week, WEEKS_BEFORE to WEEKS_AFTER around the cut
    figure_rr_d_fuel_cutoff_placebo_biweekly   cutoffs every two weeks over the same window

Each point refits the same IFLS5 sample with a different post indicator. This
is a candidate-cutoff sweep, not a lead/lag event-study regression.
"""

import pandas as pd
import pyfixest as pf
from library.config import FIGURE_OUTPUT
from library.specs import CONTROLS, fuel_shock_urban_vehicle, wave5_df
from plotnine import (
    aes,
    geom_errorbar,
    geom_hline,
    geom_point,
    geom_vline,
    ggplot,
    labs,
    scale_x_date,
    theme_minimal,
)

TRUE_DATE = pd.Timestamp("2014-11-18")
WEEKS_BEFORE = 4
WEEKS_AFTER = 14
FOCAL_TERM = "urban_vehicle_hh_ifls4:candidate_post:tmean_7d"

df = wave5_df.copy()  # Loaded from 30_analysis_table_input.parquet.
df["interview_date"] = pd.to_datetime(df["interview_datetime"]).dt.normalize()
df = df.dropna(subset=[
    "cesd_z", "tmean_7d", "urban_vehicle_hh_ifls4", "interview_date",
    "month", "year", "gadm_fullcode", "kabupaten_full_code",
    *[column.strip() for column in CONTROLS.split("+")],
])
df = df.loc[df.groupby("gadm_fullcode")["gadm_fullcode"].transform("size") > 1].copy()

formula = fuel_shock_urban_vehicle.formula.replace("post_subsidy", "candidate_post")


def weekly_cutoffs(weeks_before: int, weeks_after: int) -> list[pd.Timestamp]:
    """Cutoffs in whole-week steps from the true date, kept inside the IFLS5 field period."""
    cutoffs = [
        TRUE_DATE + pd.Timedelta(weeks=week)
        for week in range(-weeks_before, weeks_after + 1)
    ]
    # Each cutoff must leave at least one interview day on both sides.
    return [
        cutoff for cutoff in cutoffs
        if df["interview_date"].min() < cutoff <= df["interview_date"].max()
    ]


# Identification assumes interview-date heat variation is unrelated to other
# causes of CES-D conditional on demographics, month-by-year and GADM fixed
# effects. Moving the cutoff is descriptive, not evidence of a causal event path.
def estimate_cutoff(cutoff: pd.Timestamp) -> dict:
    df["candidate_post"] = (df["interview_date"] >= cutoff).astype(int)
    model = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_full_code"})
    coefficient = model.tidy().loc[FOCAL_TERM]
    return {
        "cutoff": cutoff,
        "weeks_from_true": (cutoff - TRUE_DATE).days // 7,
        "estimate": coefficient["Estimate"],
        "lower": coefficient["2.5%"],
        "upper": coefficient["97.5%"],
    }


def placebo_plot(results: pd.DataFrame, step: str):
    observed = results.loc[results["cutoff"].eq(TRUE_DATE)]
    return (
        ggplot(results, aes(x="cutoff", y="estimate"))
        + geom_hline(yintercept=0, linetype="dashed", color="gray")
        + geom_vline(xintercept=TRUE_DATE, linetype="dotted", color="#B34A3C")
        + geom_errorbar(aes(ymin="lower", ymax="upper"), width=2, color="#4C78A8")
        + geom_point(color="#4C78A8", size=1.5)
        + geom_errorbar(aes(ymin="lower", ymax="upper"), data=observed, width=2, color="#B34A3C")
        + geom_point(data=observed, color="#B34A3C", size=2)
        + scale_x_date(breaks=list(results["cutoff"]), date_labels="%d %b\n%Y")
        + labs(
            title="Fuel-shock coefficient by candidate cutoff date",
            subtitle=(
                f"CES-D z-score; cutoffs every {step}, {WEEKS_BEFORE} weeks before to "
                f"{WEEKS_AFTER} weeks after; kabupaten-clustered 95% CIs"
            ),
            x="Candidate cutoff date",
            y="Coefficient on heat × urban-vehicle × post",
            caption="Red marks the true cutoff: November 18, 2014.",
        )
        + theme_minimal(base_size=10)
    )


cutoffs = weekly_cutoffs(WEEKS_BEFORE, WEEKS_AFTER)
results = pd.DataFrame([estimate_cutoff(cutoff) for cutoff in cutoffs])
# Two-week steps reuse the weekly estimates, keeping the true date as a step.
biweekly_results = results.loc[results["weeks_from_true"] % 2 == 0]

weekly_plot = placebo_plot(results, "week")
biweekly_plot = placebo_plot(biweekly_results, "two weeks")

FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
for extension in ["pdf", "png"]:
    weekly_plot.save(
        FIGURE_OUTPUT / f"figure_rr_d_fuel_cutoff_placebo.{extension}",
        width=10, height=5, dpi=300, verbose=False,
    )
    biweekly_plot.save(
        FIGURE_OUTPUT / f"figure_rr_d_fuel_cutoff_placebo_biweekly.{extension}",
        width=10, height=5, dpi=300, verbose=False,
    )

with pd.option_context("display.width", 200, "display.precision", 4):
    print(results.to_string(index=False))
