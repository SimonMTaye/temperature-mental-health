"""Refined financial-shock variables addressing the caveats in the v1 note.

  palm_farmer_individual : sector==1 (agriculture) AND in palm region (BPS prov 11-21,61-64)
                           — finer than just "palm region", though still constrained to the
                           1-digit sector codes IFLS exposes (no KBLI sub-codes available).

  transport_share        : monthly transport spending / total monthly HH spending,
                           continuous, from b1_ks2 (ks2type=='E' = "Transportation"). Replaces
                           binary "urban" as a finer measure of fuel-subsidy exposure.

Output: data/generated/financial_shocks_v2.parquet
  pidlink, wave, palm_farmer_individual, transport_share, transport_share_q5
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RAW = Path("E:/IFLS/extracted")
OUT = PROJECT / "data" / "generated"

sys.path.insert(0, str(Path(__file__).parent))
from _sentinels import clean_money  # noqa: E402

PALM_PROVS = {11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 61, 62, 63, 64}

# Top Indonesian rubber-producing provinces (by area/output, GAPKINDO statistics):
#   16 South Sumatra (≈ 27 % of national rubber area)
#   15 Jambi (≈ 15 %)
#   12 North Sumatra (≈ 11 %)
#   61 West Kalimantan (≈ 12 %)
#   14 Riau (≈ 10 %)
# Together these five provinces account for ~ 75 % of Indonesia's rubber output.
RUBBER_PROVS = {12, 14, 15, 16, 61}

# Top Indonesian coffee-producing provinces (mix of Robusta + Arabica; ICO statistics):
#   18 Lampung (largest Robusta producer)
#   16 South Sumatra (Robusta)
#   12 North Sumatra (Mandailing, Lintong Arabica)
#   51 Bali (Kintamani Arabica)
#   73 South Sulawesi (Toraja Arabica)
#   11 Aceh (Gayo Arabica) — included though IFLS sample in Aceh is tiny
# Together these account for ~ 80 % of Indonesia's coffee output.
COFFEE_PROVS = {11, 12, 16, 18, 51, 73}


def _palm_farmer(wave: str) -> pd.DataFrame:
    folder = RAW / ("IFLS5/hh14" if wave == "IFLS5" else "IFLS4/hh07")
    tk = pd.read_stata(folder / "b3a_tk2.dta", convert_categoricals=False)
    if "tk19ab" not in tk.columns:
        return pd.DataFrame(columns=["pidlink", "wave", "agricultural"])
    # IFLS4 stores tk19ab as " 1", " 6", etc. — strip + parse
    sec = pd.to_numeric(tk["tk19ab"].astype(str).str.strip(), errors="coerce")
    tk["agricultural"] = (sec == 1).astype(int)
    out = tk[["pidlink", "agricultural"]].drop_duplicates("pidlink").copy()
    out["wave"] = wave
    return out


def _transport_share(wave: str) -> pd.DataFrame:
    folder = RAW / ("IFLS5/hh14" if wave == "IFLS5" else "IFLS4/hh07")
    hhcol = "hhid14" if wave == "IFLS5" else "hhid07"

    # Monthly non-food spending by category (clean sentinels first)
    ks2 = pd.read_stata(folder / "b1_ks2.dta", convert_categoricals=False)
    ks2["ks06"] = clean_money(ks2.ks06)
    transport = ks2[ks2.ks2type == "E"][[hhcol, "ks06"]].rename(
        columns={"ks06": "transport_spending_mo"}
    )

    # Total monthly HH spending — sum of all ks2 + all ks3 categories
    total_ks2 = ks2.groupby(hhcol)["ks06"].sum(min_count=1).rename("total_ks2_mo").reset_index()
    ks3 = pd.read_stata(folder / "b1_ks3.dta", convert_categoricals=False)
    ks3["ks08"] = clean_money(ks3.ks08)
    total_ks3 = ks3.groupby(hhcol)["ks08"].sum(min_count=1).rename("total_ks3_mo").reset_index()

    # Monthly food: weekly ks02a × 4.33
    ks0 = pd.read_stata(folder / "b1_ks0.dta", convert_categoricals=False)
    if "ks02a" in ks0.columns:
        ks0["food_mo"] = clean_money(ks0["ks02a"]) * 4.33
    else:
        ks0["food_mo"] = np.nan
    food = ks0[[hhcol, "food_mo"]]

    out = transport.drop_duplicates(hhcol).merge(total_ks2, on=hhcol)
    out = out.merge(total_ks3, on=hhcol).merge(food, on=hhcol, how="left")
    # Fill NaN components with 0 only AFTER sentinel masking — so a sentinel becomes 0
    # contribution (the household didn't report that category), not an inflated value.
    for c in ["transport_spending_mo", "total_ks2_mo", "total_ks3_mo", "food_mo"]:
        out[c] = out[c].fillna(0)
    out["total_mo"] = out.total_ks2_mo + out.total_ks3_mo + out.food_mo
    out["transport_share"] = out.transport_spending_mo / out.total_mo.replace(0, np.nan)
    out = out[(out.transport_share >= 0) & (out.transport_share <= 1)]
    out = out.rename(columns={hhcol: "hhid"})[
        ["hhid", "transport_spending_mo", "total_mo", "transport_share"]
    ]
    out["wave"] = wave
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    pf = pd.concat([_palm_farmer("IFLS4"), _palm_farmer("IFLS5")], ignore_index=True)
    print(f"sector data: {len(pf):,};  agricultural mean = {pf.agricultural.mean():.3f}")

    ts = pd.concat([_transport_share("IFLS4"), _transport_share("IFLS5")], ignore_index=True)
    print(f"transport-share rows: {len(ts):,};  median share = {ts.transport_share.median():.3f}")

    stressors = pd.read_parquet(OUT / "stressors.parquet")
    base = stressors[["pidlink", "wave", "hhid"]].drop_duplicates(["pidlink", "wave"])
    ind = pd.read_parquet(OUT / "individuals.parquet")[["pidlink", "wave", "prov_code"]]

    out = base.merge(ind, on=["pidlink", "wave"], how="left")
    out = out.merge(pf, on=["pidlink", "wave"], how="left")
    # transport-share is HH×wave level — dedup before merging to avoid m:n explosion
    ts_dedup = ts.drop_duplicates(subset=["hhid", "wave"])
    out = out.merge(ts_dedup, on=["hhid", "wave"], how="left")

    out["palm_region"] = out.prov_code.isin(PALM_PROVS).astype(int)
    out["palm_farmer_individual"] = (
        (out.agricultural.fillna(0) == 1) & (out.palm_region == 1)
    ).astype(int)
    # Household-level palm exposure: =1 if any HH member is a palm farmer.
    # Captures within-household income spillovers (spouses, adult children,
    # parents share the income shock with the working palm farmer).
    out["palm_farmer_hh"] = out.groupby(["hhid", "wave"])["palm_farmer_individual"].transform("max")
    out["rubber_region"] = out.prov_code.isin(RUBBER_PROVS).astype(int)
    out["rubber_farmer_individual"] = (
        (out.agricultural.fillna(0) == 1) & (out.rubber_region == 1)
    ).astype(int)
    out["coffee_region"] = out.prov_code.isin(COFFEE_PROVS).astype(int)
    out["coffee_farmer_individual"] = (
        (out.agricultural.fillna(0) == 1) & (out.coffee_region == 1)
    ).astype(int)

    # Quintile of transport share (within wave) — for binary heterogeneity
    out["transport_share_q5"] = out.groupby("wave")["transport_share"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1
    )
    out["high_transport_share"] = (out.transport_share_q5 == 5).astype(int)

    keep = ["pidlink", "wave", "agricultural",
            "palm_region", "palm_farmer_individual", "palm_farmer_hh",
            "rubber_region", "rubber_farmer_individual",
            "coffee_region", "coffee_farmer_individual",
            "transport_spending_mo", "total_mo", "transport_share",
            "transport_share_q5", "high_transport_share"]
    out_final = out[keep].copy()
    out_final.to_parquet(OUT / "financial_shocks_v2.parquet", index=False)

    print(f"\nwrote {len(out_final):,} rows to {OUT/'financial_shocks_v2.parquet'}")
    print("\nrefined shock prevalence by wave:")
    print(out_final.groupby("wave").agg(
        n=("pidlink", "size"),
        agricultural_pct=("agricultural", lambda s: 100 * s.fillna(0).mean()),
        palm_region_pct=("palm_region", lambda s: 100 * s.mean()),
        palm_farmer_pct=("palm_farmer_individual", lambda s: 100 * s.mean()),
        rubber_region_pct=("rubber_region", lambda s: 100 * s.mean()),
        rubber_farmer_pct=("rubber_farmer_individual", lambda s: 100 * s.mean()),
        coffee_region_pct=("coffee_region", lambda s: 100 * s.mean()),
        coffee_farmer_pct=("coffee_farmer_individual", lambda s: 100 * s.mean()),
        transport_share_med=("transport_share", "median"),
    ).round(3))


if __name__ == "__main__":
    main()
