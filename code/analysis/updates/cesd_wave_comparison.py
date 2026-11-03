"""Plot within-wave standardized CES-D distributions for IFLS4/IFLS5.

Run from the repo root:

    uv run python code/analysis/updates/cesd_wave_comparison.py

Writes three figures to output/figures/:
    cesd_wave_z_distribution.png                     all respondents
    cesd_wave_z_distribution_shocked.png             shocked vs not shocked
    cesd_wave_z_distribution_temperature_quartile.png  hot vs cool week
"""

import pandas as pd
from data.config import GENERATED_DATA
from library.config import FIGURE_OUTPUT
from library.specs import JOB_LOSS_MAIN, MAIN_TEMP_MEASURE
from plotnine import (
    aes,
    after_stat,
    facet_grid,
    facet_wrap,
    geom_histogram,
    ggplot,
    labs,
    theme_minimal,
)

df = pd.read_parquet(
    GENERATED_DATA / "30_analysis_table_input.parquet",
    columns=[
        "wave", "cesd_z", "palm_farmer_hh_ifls4", "urban_vehicle_hh_ifls4",
        "post_subsidy", JOB_LOSS_MAIN, MAIN_TEMP_MEASURE,
    ],
)
df = df.loc[df["wave"].isin(["IFLS4", "IFLS5"])]


def z_distribution_plot(df, title):
    return (
        ggplot(df.dropna(subset=["cesd_z"]), aes(x="cesd_z", y=after_stat("density")))
        + geom_histogram(binwidth=0.1, boundary=-0.05, color="white", fill="#4C78A8")
        + facet_wrap("~wave", nrow=1)
        + labs(title=title, x="Within-wave standardized CES-D score", y="Density")
        + theme_minimal(base_size=10)
    )


z_plot = z_distribution_plot(df, "CES-D standardized scores by IFLS wave")

# Shock groups follow table_b_temperature_and_shock_effects.py. Group
# indicators are IFLS4 household attributes, so IFLS5 respondents without an
# IFLS4 household record (<NA>) are dropped from the split below.
# Shocked = palm farmer, urban-vehicle household interviewed after the fuel
# subsidy cut, or job loss. Not shocked = none of the three, with every group
# indicator observed.
palm_farmer = df["palm_farmer_hh_ifls4"].eq(1)
non_palm_farmer = df["palm_farmer_hh_ifls4"].eq(0)
fuel_shocked = df["urban_vehicle_hh_ifls4"].eq(1) & df["post_subsidy"].eq(1)
fuel_known = df["urban_vehicle_hh_ifls4"].notna()
no_shock = non_palm_farmer & fuel_known & ~fuel_shocked & df[JOB_LOSS_MAIN].eq(0)
shocked = palm_farmer | fuel_shocked | df[JOB_LOSS_MAIN].eq(1)
shock_df = df.loc[shocked | no_shock].assign(
    shock_group=lambda d: pd.Categorical(
        shocked.loc[d.index].map({True: "Shocked", False: "Not shocked"}),
        categories=["Shocked", "Not shocked"],
    )
)
shock_plot = (
    z_distribution_plot(shock_df, "CES-D standardized scores by IFLS wave and shock status")
    + facet_grid("shock_group ~ wave")
)

# Hot/cool = top/bottom quartile of 7-day mean temperature within each wave.
# Raw temperature quartiles mostly sort cool highland from hot lowland
# kecamatans, so this is descriptive, not the within-kecamatan variation the
# regressions use.
temp_p25 = df.groupby("wave")[MAIN_TEMP_MEASURE].transform("quantile", 0.25)
temp_p75 = df.groupby("wave")[MAIN_TEMP_MEASURE].transform("quantile", 0.75)
hot = df[MAIN_TEMP_MEASURE].ge(temp_p75)
cool = df[MAIN_TEMP_MEASURE].le(temp_p25)
temperature_df = df.loc[hot | cool].assign(
    temperature_group=lambda d: pd.Categorical(
        hot.loc[d.index].map({True: "Hot week (≥ p75)", False: "Cool week (≤ p25)"}),
        categories=["Hot week (≥ p75)", "Cool week (≤ p25)"],
    )
)
temperature_plot = (
    z_distribution_plot(
        temperature_df,
        "CES-D standardized scores by IFLS wave and within-wave 7-day mean temperature quartile",
    )
    + facet_grid("temperature_group ~ wave")
)

FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
z_plot.save(FIGURE_OUTPUT / "cesd_wave_z_distribution.png", width=8, height=4, dpi=300)
shock_plot.save(FIGURE_OUTPUT / "cesd_wave_z_distribution_shocked.png", width=8, height=7, dpi=300)
temperature_plot.save(FIGURE_OUTPUT / "cesd_wave_z_distribution_temperature_quartile.png", width=8, height=7, dpi=300)
