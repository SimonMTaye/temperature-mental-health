"""Plot raw and within-wave standardized CES-D distributions for IFLS4/IFLS5."""

import pandas as pd
from plotnine import (
    after_stat, aes, facet_grid, facet_wrap, geom_histogram, ggplot, labs,
    theme_minimal,
)

from data.config import GENERATED_DATA
from library.config import FIGURE_OUTPUT
from library.specs import JOB_LOSS_MAIN


df = pd.read_parquet(
    GENERATED_DATA / "30_analysis_table_input.parquet",
    columns=[
        "wave", "cesd_raw", "cesd_z", "palm_farmer_hh_ifls4",
        "urban_vehicle_hh_ifls4", "post_subsidy", JOB_LOSS_MAIN,
    ],
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


def z_distribution_plot(df, title):
    return (
        ggplot(df.dropna(subset=["cesd_z"]), aes(x="cesd_z", y=after_stat("density")))
        + geom_histogram(binwidth=0.1, boundary=-0.05, color="white", fill="#4C78A8")
        + facet_wrap("~wave", nrow=1)
        + labs(title=title, x="Within-wave standardized CES-D score", y="Density")
        + theme_minimal(base_size=10)
    )


print_raw_score_percentiles(df)

raw_plot = (
    ggplot(df.dropna(subset=["cesd_raw"]), aes(x="cesd_raw", y=after_stat("density")))
    + geom_histogram(binwidth=1, boundary=-0.5, color="white", fill="#4C78A8")
    + facet_wrap("~wave", nrow=1)
    + labs(title="CES-D raw scores by IFLS wave", x="CES-D raw score (0–30)", y="Density")
    + theme_minimal(base_size=10)
)
z_plot = z_distribution_plot(df, "CES-D standardized scores by IFLS wave")

# Shock groups follow table_b_temperature_and_shock_effects.py. Group
# indicators are IFLS4 household attributes, so IFLS5 respondents without an
# IFLS4 household record (<NA>) are dropped from every split below.
palm_farmer = df["palm_farmer_hh_ifls4"].eq(1)
non_palm_farmer = df["palm_farmer_hh_ifls4"].eq(0)
fuel_shocked = df["urban_vehicle_hh_ifls4"].eq(1) & df["post_subsidy"].eq(1)
fuel_known = df["urban_vehicle_hh_ifls4"].notna()
no_shock = non_palm_farmer & fuel_known & ~fuel_shocked & df[JOB_LOSS_MAIN].eq(0)

palm_df = df.loc[palm_farmer | non_palm_farmer].assign(
    palm_group=lambda d: pd.Categorical(
        d["palm_farmer_hh_ifls4"].map({1: "Palm farmers", 0: "Non-palm farmers"}),
        categories=["Palm farmers", "Non-palm farmers"],
    )
)
palm_plot = (
    z_distribution_plot(palm_df, "CES-D standardized scores by IFLS wave and IFLS4 palm-farmer status")
    + facet_grid("palm_group ~ wave")
)
no_shock_plot = z_distribution_plot(
    df.loc[no_shock],
    "CES-D standardized scores by IFLS wave: no palm, fuel, or job-loss shock",
)

FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
raw_plot.save(FIGURE_OUTPUT / "cesd_wave_distribution.png", width=8, height=4, dpi=300)
z_plot.save(FIGURE_OUTPUT / "cesd_wave_z_distribution.png", width=8, height=4, dpi=300)
palm_plot.save(FIGURE_OUTPUT / "cesd_wave_z_distribution_palm.png", width=8, height=7, dpi=300)
no_shock_plot.save(FIGURE_OUTPUT / "cesd_wave_z_distribution_no_shock.png", width=8, height=4, dpi=300)
