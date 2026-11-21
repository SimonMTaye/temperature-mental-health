"""Appendix figure: World Bank monthly palm-oil price, 2007--2015 (continuous).

Run from the repo root:

    uv run python code/analysis/updates/figures/figure_rr_e_appendix_palm_price.py

Ported from Jingyao's archive/25_appendix_palm_price.py. The IFLS fielding
windows are now the first and last interview dates in the analysis table
rather than hard-coded dates. IFLS5 Book 3B visits were recorded until
2015-12-18, but every visit after 2015-09-29 was unsuccessful (respondent
unavailable or refused, b3b_cov result = 3), so the IFLS5 window ends in
September 2015.

The earlier draft chart had only the two IFLS-fielding sub-windows plotted
(2007:6--2008:9 and 2014:8--2015:12) with a six-year gap in between, because
the analysis only needs prices that fall inside the 3-month lag-window used
to construct the palm-shock variable. For the paper appendix we want a
continuous series so the reader sees the full price trajectory --- the 2008
commodity-boom peak and post-GFC crash, the 2010--2011 recovery, the
2012--2014 secular decline, and the 2014--2015 China-demand collapse.

The script:
  1. Reads the monthly palm-oil price (World Bank Pink Sheet, Malaysia FOB,
     USD/MT) from PALM_PRICE_FULL in data/_commodity_prices.py, the same
     series the pipeline uses for palm_price_usd_mt. It matches the "Palm oil"
     column of CMO-Historical-Data-Monthly.xlsx (Jan 2025 release) exactly.
  2. Plots the full 2007--2015 series with both IFLS fielding windows
     shaded, the long-run mean annotated, and key price events labelled.

Outputs (output/figures/ keeps the appendix_* names used by the paper):
  data/generated/palm_oil_monthly_2007_2015.csv
  output/figures/appendix_palm_price.png
  output/figures/appendix_palm_price.pdf
"""
from __future__ import annotations


import matplotlib.pyplot as plt
import pandas as pd
from data._commodity_prices import PALM_PRICE_FULL
from library.config import FIGURE_OUTPUT, PROJECT
from library.specs import ANALYSIS_INPUT

OUT_FIG = FIGURE_OUTPUT
OUT_DAT = PROJECT / "data" / "generated"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DAT.mkdir(parents=True, exist_ok=True)

CACHE_CSV = OUT_DAT / "palm_oil_monthly_2007_2015.csv"


def load_palm_prices() -> pd.DataFrame:
    """Monthly World Bank palm-oil prices, 2007--2015, dated mid-month."""
    rows = [
        (pd.Timestamp(year=year, month=month, day=15), price)
        for (year, month), price in sorted(PALM_PRICE_FULL.items())
        if 2007 <= year <= 2015
    ]
    return pd.DataFrame(rows, columns=["date", "price_usd_mt"])


def fielding_windows() -> pd.DataFrame:
    """First and last interview date per wave in the canonical analysis table."""
    interviews = pd.read_parquet(
        ANALYSIS_INPUT, columns=["wave", "interview_datetime"]
    )
    dates = interviews["interview_datetime"].dt.normalize()
    return dates.groupby(interviews["wave"]).agg(["min", "max"])


def fielding_label(wave: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{wave} fielding ({start:%b %Y} -- {end:%b %Y})"


def main() -> None:
    df = load_palm_prices()
    df.to_csv(CACHE_CSV, index=False)
    print(f"wrote {CACHE_CSV}  ({len(df)} months)")

    mean_p = df.price_usd_mt.mean()
    sd_p = df.price_usd_mt.std()
    print(f"  2007-15 mean = USD {mean_p:.0f}/MT; SD = USD {sd_p:.0f}/MT")

    fig, ax = plt.subplots(figsize=(13, 4.8), dpi=140)

    # IFLS fielding windows: first to last interview in the analysis table
    windows = fielding_windows()
    for wave, color in [("IFLS4", "#cfe3f4"), ("IFLS5", "#fbe1cf")]:
        start, end = windows.loc[wave, "min"], windows.loc[wave, "max"]
        ax.axvspan(start, end, color=color, alpha=0.55,
                   label=fielding_label(wave, start, end))

    # Long-run mean
    ax.axhline(mean_p, color="#888", linestyle=":", linewidth=1.2,
               label=f"2007--15 mean = USD {mean_p:.0f}/MT")

    # Price line
    ax.plot(df.date, df.price_usd_mt, color="#1a4f8b", linewidth=2.0,
            marker="o", markersize=3.0,
            label="Palm-oil monthly price (World Bank Pink Sheet)")

    # Annotate key events
    def _ann(yr, mo, label_text, dx_years, dy):
        row = df[df.date == pd.Timestamp(year=yr, month=mo, day=15)]
        if row.empty:
            return
        date, price = row.iloc[0].date, row.iloc[0].price_usd_mt
        ax.annotate(
            label_text,
            xy=(date, price),
            xytext=(date + pd.DateOffset(years=dx_years[0], months=dx_years[1]), dy),
            fontsize=8.5, ha="left",
            arrowprops=dict(arrowstyle="->", color="#666", lw=0.8),
        )

    _ann(2008,  3, f"Mar 2008 peak\nUSD {df[df.date==pd.Timestamp('2008-03-15')].iloc[0].price_usd_mt:.0f}",
         (0, 6), 1380)
    _ann(2008, 11, f"Nov 2008 trough\nUSD {df[df.date==pd.Timestamp('2008-11-15')].iloc[0].price_usd_mt:.0f}\n(post-GFC crash)",
         (1, 0), 380)
    _ann(2011,  2, f"Feb 2011 peak\nUSD {df[df.date==pd.Timestamp('2011-02-15')].iloc[0].price_usd_mt:.0f}",
         (0, 4), 1380)
    _ann(2015,  9, f"Sep 2015 trough\nUSD {df[df.date==pd.Timestamp('2015-09-15')].iloc[0].price_usd_mt:.0f}\n(China-demand collapse)",
         (-2, 0), 380)

    ax.set_xlabel("Month")
    ax.set_ylabel("USD / metric ton")
    ax.set_title("Palm-oil price across the inter-IFLS period and both fielding windows, 2007--2015",
                 fontsize=12.5, weight="bold", loc="left", pad=8)
    ax.set_ylim(350, 1450)
    ax.set_xlim(pd.Timestamp("2006-11-01"), pd.Timestamp("2016-03-01"))
    ax.legend(loc="upper right", fontsize=8.5, frameon=True,
              framealpha=0.95, edgecolor="#cccccc")
    ax.grid(alpha=0.3, linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out_path = OUT_FIG / f"appendix_palm_price.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=180)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
