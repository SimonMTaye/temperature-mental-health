"""Build Figure A: temperature descriptives in the IFLS analysis sample.

The figure describes interview-day temperature variation using the canonical
analysis input. The descriptives are kabupaten-date weighted: each unique
kabupaten_code x interview_date x wave observation contributes once.

Outputs:
  output/figures/figure_a_temperature_descriptives.pdf
  output/figures/figure_a_temperature_descriptives.png
"""

from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_text,
    geom_errorbar,
    geom_histogram,
    geom_point,
    ggplot,
    labs,
    scale_x_continuous,
    theme,
    theme_minimal,
)

from library.config import FIGURE_OUTPUT

PROJECT = Path(__file__).resolve().parents[3]
ANALYSIS_INPUT = PROJECT / "data" / "generated" / "30_analysis_table_input.parquet"
PDF_OUTPUT = FIGURE_OUTPUT / "figure_a_temperature_descriptives.pdf"
PNG_OUTPUT = FIGURE_OUTPUT / "figure_a_temperature_descriptives.png"

REQUIRED_COLUMNS = [
    "kabupaten_code",
    "interview_datetime",
    "wave",
    "month",
    "year",
    "tmean_c",
    "tmax_c",
    "tmin_c",
]

WAVE_ORDER = ["IFLS4", "IFLS5"]
MONTH_BREAKS = list(range(1, 13))
FIGURE_THEME = theme_minimal(base_family="DejaVu Serif", base_size=8) + theme(
    panel_grid_major_x=element_blank(),
    panel_grid_minor_x=element_blank(),
    plot_title=element_text(size=9),
    legend_position="bottom",
    legend_title=element_text(size=7),
    legend_text=element_text(size=7),
)


if __name__ == "__main__":
    FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
    df: pd.DataFrame = pd.read_parquet(ANALYSIS_INPUT)
    df = (
        df[REQUIRED_COLUMNS]
        .dropna(subset=["kabupaten_code", "interview_datetime", "wave", "tmean_c"])
        .copy()
    )
    df = df[df["tmean_c"] > 0].copy()
    df["interview_date"] = pd.to_datetime(df["interview_datetime"]).dt.normalize()
    df["month"] = df["month"].astype(int)
    df["year"] = df["year"].astype(int)
    df["kabupaten_code"] = df["kabupaten_code"].astype(int)
    df = df.drop_duplicates(["kabupaten_code", "interview_date", "wave"])

    monthly = (
        df.groupby("month", observed=True)
        .agg(
            median_tmean=("tmean_c", "median"),
            p25_tmean=("tmean_c", lambda s: s.quantile(0.25)),
            p75_tmean=("tmean_c", lambda s: s.quantile(0.75)),
            n=("tmean_c", "size"),
        )
        .reset_index()
        .sort_values("month")
    )

    kab_month_wave = (
        df.groupby(["kabupaten_code", "month", "wave"], observed=True)["tmean_c"]
        .mean()
        .unstack("wave")
    )
    if set(WAVE_ORDER).issubset(kab_month_wave.columns):
        monthly_delta = (
            kab_month_wave["IFLS5"]
            .sub(kab_month_wave["IFLS4"])
            .dropna()
            .rename("delta_tmean_c")
            .reset_index()
        )
        monthly_change = (
            monthly_delta.groupby("month", observed=True)
            .agg(
                median_change=("delta_tmean_c", "median"),
                p25_change=("delta_tmean_c", lambda s: s.quantile(0.25)),
                p75_change=("delta_tmean_c", lambda s: s.quantile(0.75)),
                n=("delta_tmean_c", "size"),
            )
            .reset_index()
            .sort_values("month")
        )
    else:
        monthly_change = pd.DataFrame(
            columns=["month", "median_change", "p25_change", "p75_change", "n"]
        )
    if monthly_change.empty:
        raise ValueError(
            "no same-month kabupaten observations are available in both waves"
        )

    distribution_panel = (
        ggplot(df, aes(x="tmean_c"))
        + geom_histogram(binwidth=0.5)
        + labs(
            title="Panel A: Distribution of temperature",
            x="Daily mean temperature (°C)",
            y="Observations",
        )
        + FIGURE_THEME
    )

    monthly_panel = (
        ggplot(monthly, aes(x="month", y="median_tmean"))
        + geom_errorbar(aes(ymin="p25_tmean", ymax="p75_tmean"), width=0.18)
        + geom_point(size=2.2)
        + scale_x_continuous(breaks=MONTH_BREAKS, limits=(0.5, 12.5))
        + labs(
            title="Panel B: Temperature across the year",
            x="Month",
            y="Daily mean temperature (°C)",
        )
        + FIGURE_THEME
    )

    monthly_change = monthly_change[monthly_change["n"] > 1]
    change_panel = (
        ggplot(monthly_change, aes(x="month", y="median_change"))
        + geom_errorbar(aes(ymin="p25_change", ymax="p75_change"), width=0.18)
        + geom_point(size=2.2)
        + scale_x_continuous(breaks=MONTH_BREAKS, limits=(0.5, 12.5))
        + labs(
            title="Panel C: Change from IFLS4 to IFLS5",
            x="Month",
            y="Change in Temperature (°C)",
        )
        + FIGURE_THEME
    )

    (distribution_panel / (monthly_panel | change_panel)).save(
        PDF_OUTPUT, width=9.0, height=8.0
    )
    (distribution_panel / (monthly_panel | change_panel)).save(
        PNG_OUTPUT, width=9.0, height=8.0, dpi=300
    )
    print(f"wrote {PDF_OUTPUT}")
    print(f"wrote {PNG_OUTPUT}")
