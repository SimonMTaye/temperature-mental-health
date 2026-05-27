"""Build IFLS kabupaten -> GADM polygon lookup.

Why kabupaten and not kecamatan?
  IFLS exposes BPS prov + kab + kec codes, but its kec codes are from a 2007/2010-era
  BPS scheme that BPS later renumbered. Only ~9% of IFLS kec codes match the current
  BPS reference. Kabupaten codes match cleanly (99%) so kabupaten is the smallest unit
  we can reliably geocode without a historical BPS code crosswalk.

Why polygon (not centroid)?
  Within a kabupaten — especially in mountainous Java — temperature varies by elevation.
  Polygon-mean over ERA5-Land (~9 km grid) gives a representative estimate, not just
  the value at one point.

Steps
-----
1. Cached BPS reference (cahyadsn/wilayah) -> kab_code -> kab_name.
2. GADM v4.1 Indonesia adm-2 polygons (already downloaded to E:/IFLS/extracted/gadm/).
3. Match by normalized name (within province ADM1 if needed for disambiguation).
4. Store polygon geometry as WKT in the parquet so it's directly usable by GEE later.
5. Fallback: unmatched kabupaten get the province polygon (flagged).

Outputs
-------
data/generated/kabupaten_polygons.parquet
  cols: kab_code, prov_code, nama_kab, nama_prov, geometry_wkt,
        centroid_lat, centroid_lon, area_km2, match_level (kab / prov)
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "data" / "generated"
GADM_PATH = Path("E:/IFLS/extracted/gadm/gadm41_IDN.gpkg")  # downloaded by 02_kecamatan attempt
BPS_CACHE = OUT / "bps_wilayah_2025.csv"


def normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"\b(kabupaten|kab\.?|kota|kotamadya|administrasi|provinsi)\b", "", s)
    s = s.replace("’", "").replace("'", "").replace("`", "")
    return re.sub(r"[^a-z0-9]+", "", s)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert BPS_CACHE.exists(), "Run 02_kecamatan_centroids first to cache the BPS lookup, or rebuild it."
    assert GADM_PATH.exists(), f"GADM file missing at {GADM_PATH}"

    # --- BPS kab reference
    bps = pd.read_csv(BPS_CACHE, dtype={"kode": str})
    bps_prov = bps[bps.kode.str.count(r"\.") == 0].copy()
    bps_prov["prov_code"] = bps_prov.kode.astype(int)
    bps_prov = bps_prov.rename(columns={"nama": "nama_prov"})[["prov_code", "nama_prov"]]

    bps_kab = bps[bps.kode.str.count(r"\.") == 1].copy()
    parts = bps_kab.kode.str.split(".")
    bps_kab["prov_code"] = parts.str[0].astype(int)
    bps_kab["kab_in_prov"] = parts.str[1].astype(int)
    bps_kab["kab_code"] = bps_kab.prov_code * 100 + bps_kab.kab_in_prov
    bps_kab["nama_kab"] = bps_kab.nama
    bps_kab["nm"] = bps_kab.nama.map(normalize)
    # BPS convention: kab_in_prov >= 71 is kota (city); below is kabupaten (regency).
    # Used to disambiguate when a normalized name (e.g. "bogor") exists as both.
    bps_kab["is_kota"] = bps_kab.kab_in_prov >= 71
    print(f"BPS kab reference: {len(bps_kab)}")

    # --- IFLS kab codes
    ind = pd.read_parquet(OUT / "individuals.parquet")
    ifls = ind[["prov_code", "kab_code"]].drop_duplicates().reset_index(drop=True)
    ifls = ifls.merge(bps_kab[["kab_code", "nama_kab", "nm", "is_kota"]], on="kab_code", how="left")
    ifls = ifls.merge(bps_prov, on="prov_code", how="left")
    print(f"IFLS unique kab: {len(ifls)};  with BPS name: {ifls.nama_kab.notna().sum()}")

    # --- GADM adm-2 (kab) and adm-1 (prov) polygons
    print("reading GADM polygons...")
    g2 = gpd.read_file(GADM_PATH, layer="ADM_ADM_2").to_crs(4326)
    g2["nm"] = g2["NAME_2"].map(normalize)
    g2["nm_prov"] = g2["NAME_1"].map(normalize)
    g1 = gpd.read_file(GADM_PATH, layer="ADM_ADM_1").to_crs(4326)
    g1["nm_prov"] = g1["NAME_1"].map(normalize)

    # Compute centroid (in equal-area projection) + area
    print("computing centroids + areas...")
    g2_eq = g2.to_crs(3857)
    g2["centroid_lat"] = g2_eq.centroid.to_crs(4326).y
    g2["centroid_lon"] = g2_eq.centroid.to_crs(4326).x
    g2["area_km2"] = g2_eq.area / 1e6
    g1_eq = g1.to_crs(3857)
    g1["centroid_lat"] = g1_eq.centroid.to_crs(4326).y
    g1["centroid_lon"] = g1_eq.centroid.to_crs(4326).x
    g1["area_km2"] = g1_eq.area / 1e6

    # --- Match IFLS kab to GADM polygon by normalized name
    # Disambiguate by (a) province (b) kab/kota type (regency vs city)
    g2 = g2.assign(is_kota_gadm=g2["ENGTYPE_2"].str.contains("City", case=False, na=False))
    g2_p = g2[["nm", "nm_prov", "is_kota_gadm", "geometry",
               "centroid_lat", "centroid_lon", "area_km2"]].rename(columns={"geometry": "geom_kab"})
    ifls_p = ifls.assign(nm_prov=ifls.nama_prov.map(normalize))

    # First pass: exact match on (nm, nm_prov, is_kota↔is_kota_gadm)
    ifls_p["is_kota_join"] = ifls_p.is_kota.fillna(False)
    matched = ifls_p.merge(
        g2_p, left_on=["nm", "nm_prov", "is_kota_join"],
        right_on=["nm", "nm_prov", "is_kota_gadm"], how="left",
    )
    # Second pass for unmatched: ignore kab/kota distinction, match on (nm, nm_prov) only
    need = matched.geom_kab.isna()
    if need.any():
        retry = ifls_p.loc[need.values, ["kab_code", "nm", "nm_prov"]].merge(
            g2_p.drop_duplicates(subset=["nm", "nm_prov"]),
            on=["nm", "nm_prov"], how="left",
        )
        for col in ["geom_kab", "centroid_lat", "centroid_lon", "area_km2"]:
            matched.loc[need.values, col] = retry[col].values
    # Third pass: drop province constraint (last-ditch)
    need = matched.geom_kab.isna()
    if need.any():
        retry = ifls_p.loc[need.values, ["kab_code", "nm"]].merge(
            g2_p.drop_duplicates(subset=["nm"]),
            on="nm", how="left",
        )
        for col in ["geom_kab", "centroid_lat", "centroid_lon", "area_km2"]:
            matched.loc[need.values, col] = retry[col].values
    # Deduplicate: keep one row per kab_code (smallest area when multiple)
    matched = matched.sort_values(["kab_code", "area_km2"]).drop_duplicates("kab_code", keep="first")

    matched["match_level"] = matched.geom_kab.notna().map(lambda x: "kab" if x else None)
    n_kab = matched.match_level.eq("kab").sum()
    print(f"matched at kab level: {n_kab} / {len(matched)}")

    # --- Fallback: province polygon for unmatched kabupaten
    g1_min = g1.assign(nm_prov_norm=g1["NAME_1"].map(normalize))[
        ["nm_prov_norm", "geometry", "centroid_lat", "centroid_lon", "area_km2"]
    ].rename(columns={
        "geometry": "geom_prov",
        "centroid_lat": "centroid_lat_p",
        "centroid_lon": "centroid_lon_p",
        "area_km2": "area_km2_p",
    })
    matched["nm_prov_norm"] = matched.nama_prov.map(normalize)
    matched = matched.merge(g1_min, on="nm_prov_norm", how="left")
    need_fb = matched.match_level.isna()
    matched.loc[need_fb, "geom_kab"] = matched.loc[need_fb, "geom_prov"]
    matched.loc[need_fb, "centroid_lat"] = matched.loc[need_fb, "centroid_lat_p"]
    matched.loc[need_fb, "centroid_lon"] = matched.loc[need_fb, "centroid_lon_p"]
    matched.loc[need_fb, "area_km2"] = matched.loc[need_fb, "area_km2_p"]
    matched.loc[need_fb & matched.geom_kab.notna(), "match_level"] = "prov"

    # --- Pack outputs
    out = matched[[
        "kab_code", "prov_code", "nama_kab", "nama_prov",
        "geom_kab", "centroid_lat", "centroid_lon", "area_km2", "match_level",
    ]].copy()
    out["geometry_wkt"] = out.geom_kab.apply(lambda g: g.wkt if g is not None else None)
    out = out.drop(columns=["geom_kab"])
    out.to_parquet(OUT / "kabupaten_polygons.parquet", index=False)

    print(f"\nwrote {len(out)} kabupaten polygons to {OUT/'kabupaten_polygons.parquet'}")
    print("match level:")
    print(out.match_level.value_counts(dropna=False))
    print("\narea (km^2) summary by match level:")
    print(out.groupby("match_level")["area_km2"].describe(percentiles=[.25, .5, .75])[
        ["count", "min", "25%", "50%", "75%", "max"]
    ])


if __name__ == "__main__":
    main()
