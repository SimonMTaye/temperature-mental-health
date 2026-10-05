"""Plot raw and within-wave standardized CES-D distributions for IFLS4/IFLS5."""

import pandas as pd
from plotnine import (
    after_stat, aes, facet_wrap, geom_histogram, ggplot, labs, theme_minimal,
)

from data.config import GENERATED_DATA
from library.config import FIGURE_OUTPUT


df = pd.read_parquet(
    GENERATED_DATA / "30_analysis_table_input.parquet",
    columns=["wave", "cesd_raw", "cesd_z"],
)
df = df.loc[df["wave"].isin(["IFLS4", "IFLS5"])]


# Raw-score percentiles from the canonical input (2026-10-04):
# wave    p1  p25  p50  p75  p95  p99
# IFLS4    0    2    3    6   11   17
# IFLS5    0    3    5    9   16   20
def print_raw_score_percentiles(df):
    table = df.groupby("wave")["cesd_raw"].quantile(
        [0.01, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).unstack()
    table.columns = ["p1", "p25", "p50", "p75", "p95", "p99"]
    print(table.to_string())


print_raw_score_percentiles(df)

raw_plot = (
    ggplot(df.dropna(subset=["cesd_raw"]), aes(x="cesd_raw", y=after_stat("density")))
    + geom_histogram(binwidth=1, boundary=-0.5, color="white", fill="#4C78A8")
    + facet_wrap("~wave", nrow=1)
    + labs(title="CES-D raw scores by IFLS wave", x="CES-D raw score (0–30)", y="Density")
    + theme_minimal(base_size=10)
)
z_plot = (
    ggplot(df.dropna(subset=["cesd_z"]), aes(x="cesd_z", y=after_stat("density")))
    + geom_histogram(binwidth=0.1, boundary=-0.05, color="white", fill="#4C78A8")
    + facet_wrap("~wave", nrow=1)
    + labs(
        title="CES-D standardized scores by IFLS wave",
        x="Within-wave standardized CES-D score", y="Density",
    )
    + theme_minimal(base_size=10)
)

FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
for extension in ["png", "pdf"]:
    raw_plot.save(FIGURE_OUTPUT / f"cesd_wave_distribution.{extension}", width=8, height=4, dpi=300)
    z_plot.save(FIGURE_OUTPUT / f"cesd_wave_z_distribution.{extension}", width=8, height=4, dpi=300)
