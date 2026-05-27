"""Appendix figure: IFLS coverage map of Indonesia.

Colours each kabupaten by the number of IFLS adults in the pooled analysis
sample (IFLS4 + IFLS5, n = 59,944) who live there. Outlines the 13 original
IFLS sampling-frame provinces (per RAND documentation) plus Banten (which
split off from West Java in 2000 and is effectively in-sample for split-off
households).

Adapted from output/make_coverage_map.py; this version is the paper-appendix
script — it writes BOTH `.png` (for the long note / web) AND `.pdf`
(vector, for LaTeX inclusion in the paper appendix).

Outputs:
  output/figures/appendix_coverage_map.png
  output/figures/appendix_coverage_map.pdf
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Original IFLS-13 sampling-frame provinces (per RAND IFLS documentation),
# plus Banten (36) which split from Jabar (32) in 2000.
IFLS13_PROVS = {12, 13, 16, 18, 31, 32, 33, 34, 35, 36, 51, 52, 63, 73}

# GADM v4.1 polygons file (Indonesia, ADM_1 + ADM_2 layers).
GADM_PATH = "E:/IFLS/extracted/gadm/gadm41_IDN.gpkg"


def normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"\b(kabupaten|kab\.?|kota|kotamadya|administrasi|provinsi)\b", "", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def main() -> None:
    # 1. Sample counts per kab from the actual analysis dataset
    df = pd.read_parquet(PROJECT / "data" / "generated" / "analysis_dataset.parquet")
    n_per_kab = df.groupby("kab_code").size().rename("n_ifls").reset_index()

    kab_lookup = pd.read_parquet(PROJECT / "data" / "generated" / "kabupaten_polygons.parquet")
    kab_lookup = kab_lookup.merge(n_per_kab, on="kab_code", how="left").fillna({"n_ifls": 0})

    # 2. GADM polygons
    g1 = gpd.read_file(GADM_PATH, layer="ADM_ADM_1").to_crs(4326)
    g2 = gpd.read_file(GADM_PATH, layer="ADM_ADM_2").to_crs(4326)
    g2["kab_norm"] = g2["NAME_2"].map(normalize)
    g2["prov_norm"] = g2["NAME_1"].map(normalize)

    # 3. Build per-(prov_norm, kab_norm) -> sample-size lookup
    kab_lookup["kab_norm"] = kab_lookup.nama_kab.map(normalize)
    kab_lookup["prov_norm"] = kab_lookup.nama_prov.map(normalize)
    sample_lookup = kab_lookup.set_index(["prov_norm", "kab_norm"])["n_ifls"].to_dict()

    g2["n_ifls"] = g2.apply(
        lambda r: sample_lookup.get((r.prov_norm, r.kab_norm), 0), axis=1
    )

    # 4. Identify fallback kab (couldn't be matched at the kab level, only province)
    fb_kab = kab_lookup[kab_lookup.match_level == "prov"]
    fb_set = set(zip(fb_kab.prov_norm, fb_kab.kab_norm))
    g2["is_fallback"] = g2.apply(
        lambda r: (r.prov_norm, r.kab_norm) in fb_set, axis=1
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
    g1["prov_norm"] = g1["NAME_1"].map(normalize)
    bps_to_norm = {
        12: "sumaterautara", 13: "sumaterabarat", 16: "sumateraselatan", 18: "lampung",
        31: "dkijakarta", 32: "jawabarat", 33: "jawatengah", 34: "daerahistimewayogyakarta",
        35: "jawatimur", 36: "banten", 51: "bali", 52: "nusatenggarabarat",
        63: "kalimantanselatan", 73: "sulawesiselatan",
    }
    ifls13_norms = set(bps_to_norm.values())
    g1_ifls = g1[g1.prov_norm.isin(ifls13_norms)]
    g1_ifls.boundary.plot(ax=ax, edgecolor="#1d6f42", linewidth=1.6, alpha=0.90)

    # Crop to IFLS-13 region (Sumatra in west to Sulawesi/NTT in east).
    # Excludes Maluku and Papua where IFLS has essentially no sample.
    ax.set_xlim(94.5, 126.5)
    ax.set_ylim(-11.0, 6.5)
    ax.set_aspect("equal")

    # Sample-size & outline counts
    n_kab_in_sample = (g2.n_ifls > 0).sum()
    n_kab_substantial = (g2.n_ifls >= 50).sum()
    n_fb = int(g2.is_fallback.sum())

    # Legends — placed OUTSIDE the map (right side) so they don't cover any polygons
    legend_elements = [
        Patch(facecolor=c, edgecolor="#777", linewidth=0.3, label=f"{lab} adults")
        for c, lab in zip(bin_colors, bin_labels)
    ]
    outline_elements = [
        Line2D([0], [0], color="#1d6f42", linewidth=1.6,
               label="Original IFLS-13 sampling-frame\nprovince ($+$ Banten)"),
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

    ax.set_title(
        f"IFLS coverage  --  {n_kab_in_sample} kabupaten with at least 1 sampled adult, "
        f"{n_kab_substantial} with $\\geq$ 50 adults",
        fontsize=12.5, loc="left", weight="bold", pad=10,
    )
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
    print(f"  Fallback (province-mean) kab: {n_fb}")
    print()
    print("Sample distribution by province (top 30):")
    by_prov = df.merge(
        kab_lookup[["kab_code", "nama_prov"]], on="kab_code", how="left"
    ).groupby("nama_prov").size().sort_values(ascending=False)
    print(by_prov.head(30))


if __name__ == "__main__":
    main()
