"""Build Figure B: non-linear past-30-day temperature effects.

Outputs:
  output/figures/figure_b_temperature_nonlinear_quintile.pdf
  output/figures/figure_b_temperature_nonlinear_quintile.png
  output/figures/figure_b_temperature_nonlinear_decile.pdf
  output/figures/figure_b_temperature_nonlinear_decile.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_text,
    geom_errorbar,
    geom_hline,
    geom_point,
    ggplot,
    labs,
    scale_x_discrete,
    theme,
    theme_minimal,
)

from library.caching import run_regression_with_caching
from library.config import FIGURE_OUTPUT
from library.render import RegressionSpec
from library.specs import CONTROLS, FE_WAVE

PROJECT = Path(__file__).resolve().parents[3]
ANALYSIS_INPUT = PROJECT / "data" / "generated" / "30_analysis_table_input.parquet"

QUINTILE_PDF_OUTPUT = FIGURE_OUTPUT / "figure_b_temperature_nonlinear_quintile.pdf"
QUINTILE_PNG_OUTPUT = FIGURE_OUTPUT / "figure_b_temperature_nonlinear_quintile.png"
DECILE_PDF_OUTPUT = FIGURE_OUTPUT / "figure_b_temperature_nonlinear_decile.pdf"
DECILE_PNG_OUTPUT = FIGURE_OUTPUT / "figure_b_temperature_nonlinear_decile.png"

BASE_COLUMNS = [
    "cesd_z",
    "age",
    "female",
    "edu_yrs",
    "married",
    "widowed",
    "month",
    "year",
    "wave",
    "gadm_fullcode",
    "kabupaten_full_code",
]

FIGURE_THEME = theme_minimal(base_family="DejaVu Serif", base_size=8) + theme(
    panel_grid_major_x=element_blank(),
    panel_grid_minor_x=element_blank(),
    plot_title=element_text(size=9),
    legend_position="none",
    axis_text_x=element_text(rotation=45, hjust=1),
)


def temperature_bin_columns(family: str) -> list[str]:
    return [
        f"temp_days_past30_{family}{index}" for index in range(1, bin_count(family) + 1)
    ]


def bin_count(family: str) -> int:
    if family == "q":
        return 5
    if family == "d":
        return 10
    raise ValueError(f"unknown temperature bin family: {family}")


def bin_label(family: str, term: str) -> str:
    label = term.removeprefix(f"temp_days_past30_{family}").upper()
    return f"{family.upper()}{label}"


def load_analysis_frame(bin_columns: list[str]) -> pd.DataFrame:
    required_columns = BASE_COLUMNS + bin_columns
    df = pd.read_parquet(ANALYSIS_INPUT, columns=required_columns)
    df["ifls5"] = df["wave"] == "IFLS5"
    return df.dropna(
        subset=[col for col in required_columns if col != "wave"] + ["ifls5"]
    )


def estimate_temperature_bin_effects(
    df: pd.DataFrame,
    bin_columns: list[str],
    *,
    title: str,
) -> pd.DataFrame:
    plot_data = pd.DataFrame(
        {
            "term": bin_columns,
            "total_days": [df[col].sum() for col in bin_columns],
        }
    )
    reference_term = plot_data.loc[plot_data["total_days"].idxmax(), "term"]
    included_terms = [col for col in bin_columns if col != reference_term]

    # Identification assumes daily weather variation around interview timing is
    # as-good-as-random after controls and community, month, year, and wave FEs.
    spec = RegressionSpec(
        title=title,
        formula=f"cesd_z ~ {' + '.join(included_terms)} + {CONTROLS} | {FE_WAVE}",
        df=df,
        tags=frozenset(["past30-temperature-bin-effect"]),
        show_terms=frozenset(included_terms),
    )
    coef_table = run_regression_with_caching(spec).coef_table
    missing_terms = sorted(set(included_terms).difference(coef_table.index.astype(str)))
    if missing_terms:
        raise ValueError(f"missing past-30-day bin coefficients: {missing_terms}")

    plot_data["estimate"] = 0.0
    plot_data["ci95l"] = np.nan
    plot_data["ci95u"] = np.nan
    for term in included_terms:
        term_mask = plot_data["term"] == term
        plot_data.loc[term_mask, ["estimate", "ci95l", "ci95u"]] = [
            coef_table.loc[term, "b"],
            coef_table.loc[term, "ci95l"],
            coef_table.loc[term, "ci95u"],
        ]
    plot_data["reference"] = plot_data["term"] == reference_term
    return plot_data


def make_temperature_bin_plot(
    family: str,
    *,
    title: str,
    x_label: str,
) -> ggplot:
    bin_columns = temperature_bin_columns(family)
    df = load_analysis_frame(bin_columns)
    plot_data = estimate_temperature_bin_effects(df, bin_columns, title=title)
    plot_data["label"] = [bin_label(family, term) for term in plot_data["term"]]
    label_order = plot_data["label"].tolist()
    reference_label = plot_data.loc[plot_data["reference"], "label"].iat[0]
    errorbar_data = plot_data.dropna(subset=["ci95l", "ci95u"])

    return (
        ggplot(plot_data, aes(x="label", y="estimate"))
        + geom_hline(yintercept=0, color="#9ca3af", size=0.5)
        + geom_errorbar(
            aes(ymin="ci95l", ymax="ci95u"),
            data=errorbar_data,
            width=0.12,
            color="#2563eb",
        )
        + geom_point(color="#2563eb", size=2)
        + scale_x_discrete(limits=label_order)
        + labs(
            title=title,
            x=x_label,
            y="Effect on CES-D z-score",
            caption=f"Reference bin: {reference_label}",
        )
        + FIGURE_THEME
    )


if __name__ == "__main__":
    FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)

    quintile_plot = make_temperature_bin_plot(
        "q",
        title="Past 30-Day Temperature Effects by Quintile",
        x_label="Past-30-day mean temperature bin",
    )
    quintile_plot.save(QUINTILE_PDF_OUTPUT, width=6.5, height=4.0)
    quintile_plot.save(QUINTILE_PNG_OUTPUT, width=6.5, height=4.0, dpi=300)

    decile_plot = make_temperature_bin_plot(
        "d",
        title="Past 30-Day Temperature Effects by Decile",
        x_label="Past-30-day mean temperature bin",
    )
    decile_plot.save(DECILE_PDF_OUTPUT, width=7.0, height=4.0)
    decile_plot.save(DECILE_PNG_OUTPUT, width=7.0, height=4.0, dpi=300)

    print(f"wrote {QUINTILE_PDF_OUTPUT}")
    print(f"wrote {QUINTILE_PNG_OUTPUT}")
    print(f"wrote {DECILE_PDF_OUTPUT}")
    print(f"wrote {DECILE_PNG_OUTPUT}")
