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
1. Cached BPS reference (cahyadsn/wilayah) -> kabupaten_code -> kab_name.
2. GADM v4.1 Indonesia adm-2 polygons (already downloaded to E:/IFLS/extracted/gadm/).
3. Match by normalized name (within province ADM1 if needed for disambiguation).
4. Store polygon geometry as WKT in the parquet so it's directly usable by GEE later.
5. Fallback: unmatched kabupaten get the province polygon (flagged).

Outputs
-------
data/generated/kabupaten_polygons.parquet
  cols: kabupaten_code, province_code, nama_kab, nama_prov, geometry_wkt,
        centroid_lat, centroid_lon, area_km2, match_level (kabupaten / province)
"""

import geopandas as gpd
import pandas as pd
import importlib

from config import GADM_PATH, GENERATED_DATA
from log import log

G3 = gpd.read_file(GADM_PATH, layer="ADM_ADM_3").to_crs(4326)
G2 = gpd.read_file(GADM_PATH, layer="ADM_ADM_2").to_crs(4326)
G1 = gpd.read_file(GADM_PATH, layer="ADM_ADM_1").to_crs(4326)
# TODO: Double check the EPSG codes (geographic coordinate math stuff) used below
G3_eq = G3.to_crs(3857)


# Track unmatched records at each level for debugging and fallback logic
UNMATCHED = []
UNMATCHED_L2 = []
UNMATCHED_L1 = []


def map_to_geometry(row) -> pd.Series:
    """
    Return geometry for the given GADM code.

    Progressively matches code to
    1. ADM3 (kecamatan) polygons if possible, averaging over multiple if needed for IFLS4 boundary changes.
    2. ADM2 (kabupaten) polygon if no ADM3 match.
    3. ADM1 (province) polygon if no ADM2 match.

    """
    gadm_code = str(row["gadm_fullcode"])
    kec_codes = gadm_code.split(",")
    polygons = G3[G3.CC_3.isin(kec_codes)]
    match_level = "kecamatan"
    if len(polygons) == 0:
        # raise ValueError(
        #     f"No geometries found for:\n\t GADM code {gadm_code} Wave: {row['wave']} Kabupaten: {row['kabupaten_code']} PID: {row['hhid']}"
        # )
        log(
            f"No geometries found for:\n\t GADM code {gadm_code} Wave: {row['wave']} Kabupaten: {row['kabupaten_code']} PID: {row['hhid']}",
            "WARNING",
        )
        global UNMATCHED, UNMATCHED_L2
        UNMATCHED.append(gadm_code)
        polygons = G2[G2.CC_2 == gadm_code[:4]]
        match_level = "kabupaten"
        if len(polygons) == 0:
            # print(
            #     f"No kabupaten geometry found for:\n\t GADM code {gadm_code} Wave: {row['wave']} Kabupaten: {row['kabupaten_code']} PID: {row['hhid']}"
            # )
            UNMATCHED_L2.append(gadm_code)
            if len(polygons) == 0:
                polygons = G1[G1.CC_1 == gadm_code[:2]]
                match_level = "province"
                if len(polygons) == 0:
                    # print(f"No province geometry found for:\n\t GADM code {gadm_code}")
                    UNMATCHED_L1.append(gadm_code)
                    return pd.Series({"geometry_wkt": None, "match_level": "unmatched"})
    dissolved = polygons.geometry.union_all().wkt
    return pd.Series({"geometry_wkt": dissolved, "match_level": match_level})


def main() -> None:
    first_module = importlib.import_module("01_extract_individuals")
    geo_ifls4 = first_module.parse_geo_codes_ifls4()
    geo_ifls5 = first_module.parse_geo_codes_ifls5()
    geo_both = pd.concat([geo_ifls4, geo_ifls5], ignore_index=True)
    geo_both["gadm_fullcode"] = geo_both["gadm_fullcode"].astype(str)
    geometry_matches = geo_both.apply(map_to_geometry, axis=1)
    geo_both = pd.concat([geo_both, geometry_matches], axis=1)
    log(f"Unmatched records at L3: {len(UNMATCHED)} / {len(geo_both)}")
    log(f"Unmatched records at L2: {len(UNMATCHED_L2)} / {len(geo_both)}")
    log(f"Unmatched records at L1: {len(UNMATCHED_L1)} / {len(geo_both)}")
    log("match level counts:", "DEBUG")
    log(geo_both["match_level"].value_counts(dropna=False), "DEBUG")
    geo_both.to_parquet(GENERATED_DATA / "02_kabupaten_polygons.parquet", index=False)


if __name__ == "__main__":
    main()
