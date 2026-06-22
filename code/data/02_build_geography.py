"""Build IFLS GADM 3.6 geography -> polygon lookup.

This v2 keeps the original output contract but reads GADM 3.6 layers, whose
GeoPackage layers are named ``gadm36_IDN_1``/``2``/``3`` rather than
``ADM_ADM_1``/``2``/``3``. Some GADM 3.6 CC_* admin codes appear on multiple
adjacent polygons due to spelling or segmentation quirks, so lookup geometries
are dissolved by code before matching IFLS geography.

Outputs
-------
data/generated/02_kabupaten_polygons.parquet
  cols: gadm_fullcode, province_code, geometry_wkt, match_level
"""

import importlib

import geopandas as gpd
import pandas as pd
from shapely import union_all

from data.config import GADM_PATH, GENERATED_DATA
from library.log import log


ADM1_LAYER = "gadm36_IDN_1"
ADM2_LAYER = "gadm36_IDN_2"
ADM3_LAYER = "gadm36_IDN_3"


def read_gadm_layer(layer: str) -> gpd.GeoDataFrame:
    """Read a GADM 3.6 layer in WGS84 coordinates."""
    return gpd.read_file(GADM_PATH, layer=layer).to_crs(4326)


def build_geometry_lookup(
    geographies: gpd.GeoDataFrame, code_column: str
) -> dict[str, object]:
    """Return one dissolved geometry per non-missing GADM code."""
    coded = geographies.loc[
        geographies[code_column].notna(), [code_column, "geometry"]
    ].copy()
    coded[code_column] = coded[code_column].astype(str)
    coded = coded[coded[code_column] != "NA"].copy()

    lookup = {}
    for code, group in coded.groupby(code_column, sort=False):
        geometries = list(group.geometry)
        lookup[code] = geometries[0] if len(geometries) == 1 else union_all(geometries)
    return lookup


G3_BY_CODE = build_geometry_lookup(read_gadm_layer(ADM3_LAYER), "CC_3")
G2_BY_CODE = build_geometry_lookup(read_gadm_layer(ADM2_LAYER), "CC_2")
G1_BY_CODE = build_geometry_lookup(read_gadm_layer(ADM1_LAYER), "CC_1")


def map_to_geometry(gadm_code: str) -> dict[str, str | None]:
    """
    Return geometry for the given GADM code.

    Progressively matches code to
    1. ADM3 (kecamatan) polygons if possible.
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
    return {"geometry_wkt": geometry.wkt, "match_level": match_level}  # ty:ignore[unresolved-attribute]


def build_geometry_matches(gadm_codes: pd.Series) -> pd.DataFrame:
    """Resolve each distinct GADM code once and return a merge-ready lookup."""
    unique_codes = list(gadm_codes.drop_duplicates())

    def build_record(gadm_code: str) -> dict[str, str | None]:
        return {"gadm_fullcode": gadm_code, **map_to_geometry(gadm_code)}

    records = [build_record(gadm_code) for gadm_code in unique_codes]
    return pd.DataFrame.from_records(records)


def main() -> None:
    first_module = importlib.import_module("data.01_extract_individuals")
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
