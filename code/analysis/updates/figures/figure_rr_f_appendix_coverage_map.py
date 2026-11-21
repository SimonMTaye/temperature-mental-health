"""Appendix figure: IFLS coverage map of Indonesia.

Run from the repo root:

    uv run python code/analysis/updates/figures/figure_rr_f_appendix_coverage_map.py

Ported from Jingyao's archive/24_appendix_coverage_map.py. Counts now come
from the canonical analysis table and join to GADM 3.6 by BPS code
(kabupaten_full_code = CC_2, province_code = CC_1) instead of by name.
Needs data/raw/gadm36_IDN.gpkg (data.config.GADM_PATH).

Colours each kabupaten by the number of IFLS person-wave observations in the
pooled analysis sample (IFLS4 + IFLS5) who live there. Outlines the 13 original
IFLS sampling-frame provinces (per RAND documentation) plus Banten (which
split off from West Java in 2000 and is effectively in-sample for split-off
households).

Adapted from output/make_coverage_map.py; this version is the paper-appendix
script — it writes BOTH `.png` (for the long note / web) AND `.pdf`
(vector, for LaTeX inclusion in the paper appendix).

Outputs (output/figures/ keeps the appendix_* name used by the paper):
  output/figures/appendix_coverage_map.png
  output/figures/appendix_coverage_map.pdf
"""
from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from data.config import GADM_PATH
from library.config import FIGURE_OUTPUT
from library.specs import ANALYSIS_INPUT
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = FIGURE_OUTPUT
OUT.mkdir(parents=True, exist_ok=True)

# Original IFLS-13 sampling-frame provinces (per RAND IFLS documentation),
# plus Banten (36) which split from Jabar (32) in 2000.
IFLS13_PROVS = {12, 13, 16, 18, 31, 32, 33, 34, 35, 36, 51, 52, 63, 73}

# Kabupaten created after GADM 3.6's boundaries, mapped to the kabupaten they
# split from so their observations land on a polygon.
NEW_KABUPATEN_PARENT = {
    1612: 1603,  # Penukal Abab Lematang Ilir (2013) <- Muara Enim
    3218: 3207,  # Pangandaran (2012) <- Ciamis
    7606: 7602,  # Mamuju Tengah (2013) <- Mamuju
}


def read_gadm_layer(layer: str) -> gpd.GeoDataFrame:
    """Read a GADM 3.6 layer in WGS84 with its BPS code as an integer."""
    g = gpd.read_file(GADM_PATH, layer=layer).to_crs(4326)
    code = layer[-1]
    g["bps_code"] = pd.to_numeric(g[f"CC_{code}"], errors="coerce")
    return g


def main() -> None:
    # 1. Sample counts per kabupaten from the canonical analysis table
    df = pd.read_parquet(
        ANALYSIS_INPUT, columns=["kabupaten_full_code", "province_code"]
    )
    kabupaten = df["kabupaten_full_code"].replace(NEW_KABUPATEN_PARENT)
    n_per_kab = kabupaten.value_counts()

    # 2. GADM polygons, coloured by BPS-code match
    g1 = read_gadm_layer("gadm36_IDN_1")
    g2 = read_gadm_layer("gadm36_IDN_2")
    g2["n_ifls"] = g2["bps_code"].map(n_per_kab).fillna(0)
    unmatched = n_per_kab[~n_per_kab.index.isin(g2["bps_code"])]
    print(
        f"kabupaten in sample without a GADM CC_2 match: {len(unmatched)} "
        f"({unmatched.sum():,} observations)"
    )

    # 5. Colour scheme: sample size -> blue intensity
    bins = [0, 1, 50, 200, 500, 1000, 5000]
    bin_colors = ["#f0f0f0", "#dceaf6", "#a8c8e1", "#5a9bcc", "#2c7bb6", "#08519c"]
    bin_labels = ["0", "1--49", "50--199", "200--499", "500--999", "1000+"]
    g2["bin_idx"] = pd.cut(g2.n_ifls, bins=bins, labels=False, include_lowest=True, right=False)
    g2["color"] = g2["bin_idx"].apply(lambda i: bin_colors[int(i)] if pd.notna(i) else bin_colors[0])

    # 6. Plot — focus on the IFLS-13 region (Sumatra to Sulawesi); Maluku/Papua
    # contribute essentially zero observations to the sample and dominate the
    # figure if included.
    fig, ax = plt.subplots(1, 1, figsize=(13, 5.0), dpi=160)

    # All GADM kab, coloured by sample size
    g2.plot(ax=ax, color=g2["color"], edgecolor="white", linewidth=0.15)

    # Province outlines (light grey)
    g1.boundary.plot(ax=ax, edgecolor="#888", linewidth=0.4)

    # Highlight original IFLS-13 provinces + Banten with thick green outline
    g1_ifls = g1[g1["bps_code"].isin(IFLS13_PROVS)]
    g1_ifls.boundary.plot(ax=ax, edgecolor="#1d6f42", linewidth=1.6, alpha=0.90)

    # Crop to IFLS-13 region (Sumatra in west to Sulawesi/NTT in east).
    # Excludes Maluku and Papua where IFLS has essentially no sample.
    ax.set_xlim(94.5, 126.5)
    ax.set_ylim(-11.0, 6.5)
    ax.set_aspect("equal")

    # Sample-size & outline counts
    n_kab_in_sample = (g2.n_ifls > 0).sum()
    n_kab_substantial = (g2.n_ifls >= 50).sum()

    # Legends — placed OUTSIDE the map (right side) so they don't cover any polygons
    legend_elements = [
        Patch(facecolor=c, edgecolor="#777", linewidth=0.3, label=f"{lab} adults")
        for c, lab in zip(bin_colors, bin_labels)
    ]
    outline_elements = [
        Line2D([0], [0], color="#1d6f42", linewidth=1.6,
               label="IFLS sampling-frame provinces"),
    ]
    leg1 = ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(1.005, 1.0),
        title="IFLS adults per\nkabupaten",
        fontsize=8.5, title_fontsize=9, frameon=True, framealpha=0.95,
        edgecolor="#cccccc", labelspacing=0.35, handlelength=1.4,
    )
    ax.legend(
        handles=outline_elements, loc="lower left", bbox_to_anchor=(1.005, 0.0),
        fontsize=8.5, frameon=True, framealpha=0.95, edgecolor="#cccccc",
    )
    ax.add_artist(leg1)

    # No in-plot title/caption (the figure caption lives in the LaTeX float).
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out_path = OUT / f"appendix_coverage_map.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=180)
        print(f"wrote {out_path}")

    print(f"\n  Kab with >= 1 IFLS adult: {n_kab_in_sample}")
    print(f"  Kab with >= 50 IFLS adults: {n_kab_substantial}")
    print()
    print("Sample distribution by province (top 30):")
    province_names = g1.set_index("bps_code")["NAME_1"]
    by_prov = (
        df["province_code"].map(province_names).value_counts().head(30)
    )
    print(by_prov)


if __name__ == "__main__":
    main()
