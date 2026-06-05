"""Build IFLS GADM geography -> polygon lookup.

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
1. Parse IFLS4 and IFLS5 household geography from the individual extraction step.
2. Resolve each distinct GADM geography code to ADM3, then ADM2, then ADM1 geometry.
3. Store polygon geometry as WKT in the parquet so it's directly usable by GEE later.
4. Fallback: unmatched kecamatan get the kabupaten or province polygon (flagged).

Outputs
-------
data/generated/02_kabupaten_polygons.parquet
  cols: gadm_fullcode, province_code, geometry_wkt, match_level
"""

import geopandas as gpd
import pandas as pd
import importlib
from shapely import union_all

from config import GADM_PATH, GENERATED_DATA
from log import log

G3 = gpd.read_file(GADM_PATH, layer="ADM_ADM_3").to_crs(4326)
G2 = gpd.read_file(GADM_PATH, layer="ADM_ADM_2").to_crs(4326)
G1 = gpd.read_file(GADM_PATH, layer="ADM_ADM_1").to_crs(4326)


G3_BY_CODE = dict(zip(G3.CC_3.astype(str), G3.geometry, strict=False))
G2_BY_CODE = dict(zip(G2.CC_2.astype(str), G2.geometry, strict=False))
G1_BY_CODE = dict(zip(G1.CC_1.astype(str), G1.geometry, strict=False))


def map_to_geometry(gadm_code: str) -> dict[str, str | None]:
    """
    Return geometry for the given GADM code.

    Progressively matches code to
    1. ADM3 (kecamatan) polygons if possible, averaging over multiple if needed for IFLS4 boundary changes.
    2. ADM2 (kabupaten) polygon if no ADM3 match.
    3. ADM1 (province) polygon if no ADM2 match.

    """
    kec_codes = gadm_code.split(",")
    polygons = [G3_BY_CODE[code] for code in kec_codes if code in G3_BY_CODE]
    match_level = "kecamatan"
    if len(polygons) == 0:
        polygons = [G2_BY_CODE[gadm_code[:4]]] if gadm_code[:4] in G2_BY_CODE else []
        match_level = "kabupaten"
        if len(polygons) == 0:
            polygons = (
                [G1_BY_CODE[gadm_code[:2]]] if gadm_code[:2] in G1_BY_CODE else []
            )
            match_level = "province"
            if len(polygons) == 0:
                log(f"No geometry found for GADM code {gadm_code}", "WARNING")
                return {"geometry_wkt": None, "match_level": "unmatched"}
    geometry = polygons[0] if len(polygons) == 1 else union_all(polygons)
    return {"geometry_wkt": geometry.wkt, "match_level": match_level}


def build_geometry_matches(gadm_codes: pd.Series) -> pd.DataFrame:
    """Resolve each distinct GADM code once and return a merge-ready lookup."""
    unique_codes = list(gadm_codes.drop_duplicates())

    def build_record(gadm_code: str) -> dict[str, str | None]:
        return {"gadm_fullcode": gadm_code, **map_to_geometry(gadm_code)}

    records = [build_record(gadm_code) for gadm_code in unique_codes]
    return pd.DataFrame.from_records(records)


def main() -> None:
    first_module = importlib.import_module("01_extract_individuals")
    geo_ifls4 = first_module.parse_geo_codes_ifls4()
    geo_ifls5 = first_module.parse_geo_codes_ifls5()
    geo_both = pd.concat([geo_ifls4, geo_ifls5], ignore_index=True)
    geo_both = geo_both.drop_duplicates(subset=["hhid", "wave"], keep="first")
    geo_keys = geo_both[["gadm_fullcode", "province_code"]].drop_duplicates(
        "gadm_fullcode"
    )
    geometry_matches = build_geometry_matches(geo_keys["gadm_fullcode"])
    geo_keys = geo_keys.merge(geometry_matches, on="gadm_fullcode", how="left")
    geo_both = geo_both.merge(
        geo_keys[["gadm_fullcode", "match_level"]], on="gadm_fullcode", how="left"
    )
    log(
        f"Unmatched records at L3: {(geo_both['match_level'] != 'kecamatan').sum()} / {len(geo_both)}"
    )
    log(
        f"Unmatched records at L2: {geo_both['match_level'].isin(['province', 'unmatched']).sum()} / {len(geo_both)}"
    )
    log(
        f"Unmatched records at L1: {(geo_both['match_level'] == 'unmatched').sum()} / {len(geo_both)}"
    )
    log("match level counts:", "DEBUG")
    log(geo_both["match_level"].value_counts(dropna=False), "DEBUG")
    geo_keys.to_parquet(GENERATED_DATA / "02_kabupaten_polygons.parquet", index=False)


if __name__ == "__main__":
    main()
