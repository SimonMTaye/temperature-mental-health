"""Plot fuel-shock coefficients for cutoffs within one month of November 18, 2014.

Each point refits the same IFLS5 sample with a different post indicator. This
is a candidate-cutoff sweep, not a lead/lag event-study regression.
"""

import pandas as pd
import pyfixest as pf
from plotnine import (
    aes, geom_errorbar, geom_hline, geom_point, geom_vline, ggplot,
    labs, scale_x_date, theme_minimal,
)

from library.config import FIGURE_OUTPUT
from library.specs import CONTROLS, fuel_shock_urban_vehicle, wave5_df


df = wave5_df.copy()  # Loaded from 30_analysis_table_input.parquet.
df["interview_date"] = pd.to_datetime(df["interview_datetime"]).dt.normalize()
df = df.dropna(subset=[
    "cesd_z", "tmean_7d", "urban_vehicle_hh_ifls4", "interview_date",
    "month", "year", "gadm_fullcode", "kabupaten_full_code",
    *[column.strip() for column in CONTROLS.split("+")],
])
df = df.loc[df.groupby("gadm_fullcode")["gadm_fullcode"].transform("size") > 1].copy()

true_date = pd.Timestamp("2014-11-18")
# The first candidate must leave at least one interview day before the cutoff.
start = max(true_date - pd.DateOffset(months=1), df["interview_date"].min() + pd.Timedelta(days=1))
end = min(true_date + pd.DateOffset(months=1), df["interview_date"].max())
formula = fuel_shock_urban_vehicle.formula.replace("post_subsidy", "candidate_post")

# Identification assumes interview-date heat variation is unrelated to other
# causes of CES-D conditional on demographics, month-by-year and GADM fixed
# effects. Moving the cutoff is descriptive, not evidence of a causal event path.
results = []
for cutoff in pd.date_range(start, end):
    df["candidate_post"] = (df["interview_date"] >= cutoff).astype(int)
    model = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_full_code"})
    coefficient = model.tidy().loc["urban_vehicle_hh_ifls4:candidate_post:tmean_7d"]
    results.append({
        "cutoff": cutoff,
        "estimate": coefficient["Estimate"],
        "lower": coefficient["2.5%"],
        "upper": coefficient["97.5%"],
    })
results = pd.DataFrame(results)
observed = results.loc[results["cutoff"].eq(true_date)]

plot = (
    ggplot(results, aes(x="cutoff", y="estimate"))
    + geom_hline(yintercept=0, linetype="dashed", color="gray")
    + geom_vline(xintercept=true_date, linetype="dotted", color="#B34A3C")
    + geom_errorbar(aes(ymin="lower", ymax="upper"), width=0.5, color="#4C78A8")
    + geom_point(color="#4C78A8", size=1.5)
    + geom_errorbar(aes(ymin="lower", ymax="upper"), data=observed, width=0.5, color="#B34A3C")
    + geom_point(data=observed, color="#B34A3C", size=2)
    + scale_x_date(date_breaks="1 week", date_labels="%d %b", limits=(start, end))
    + labs(
        title="Fuel-shock coefficient by candidate cutoff date",
        subtitle="CES-D z-score; kabupaten-clustered 95% confidence intervals",
        x="Candidate cutoff date (2014)",
        y="Coefficient on heat × urban-vehicle × post",
        caption="Red marks the true cutoff: November 18, 2014.",
    )
    + theme_minimal(base_size=10)
)
FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
for extension in ["png", "pdf"]:
    plot.save(FIGURE_OUTPUT / f"fuel_cutoff_placebo.{extension}", width=8, height=5, dpi=300)
