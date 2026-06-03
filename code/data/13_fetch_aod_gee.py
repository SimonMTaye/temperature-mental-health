"""Pull MODIS Terra monthly Aerosol Optical Depth (AOD) per IFLS geography.

AOD at 550 nm is the workhorse satellite proxy for PM2.5 / haze severity. The
2015 Indonesian peat-fire haze (driven by El Niño + drained peatland fires) shows
up as a 0.5+ AOD spike across Sumatra and Kalimantan in Sep-Nov 2015.

Source: MODIS/061/MOD08_M3, band Aerosol_Optical_Depth_Land_Ocean_Mean_Mean.

Output: data/generated/aod_monthly_kab.parquet
  cols: gadm_fullcode, province_code, match_level, year, month, aod
"""

import time

import ee
import pandas as pd
import shapely.wkt

from config import GENERATED_DATA, GEE_PROEJCT_ID
from _schemas import AOD_MONTHLY_SCHEMA

BASE_WINDOWS = [
    ("2007-06-01", "2008-06-01"),  # IFLS4 part 1
    ("2008-06-01", "2008-09-01"),  # IFLS4 part 2
    ("2014-08-01", "2015-08-01"),  # IFLS5 pre-haze
    ("2015-08-01", "2016-01-01"),  # IFLS5 haze + tail
]
BATCH_MONTHS = 2


def init_gee() -> None:
    ee.Initialize(project=GEE_PROEJCT_ID)


def shapely_to_ee(g) -> ee.Geometry:
    g = g.simplify(
        0.05, preserve_topology=True
    )  # MODIS is ~1° anyway, simplify aggressively
    return ee.Geometry(g.__geo_interface__, opt_geodesic=False, opt_evenOdd=True)


def load_geographies() -> pd.DataFrame:
    geographies = pd.read_parquet(GENERATED_DATA / "02_kabupaten_polygons.parquet")
    required = ["gadm_fullcode", "geometry_wkt", "province_code", "match_level"]
    missing = [col for col in required if col not in geographies.columns]
    if missing:
        raise ValueError(f"02_kabupaten_polygons.parquet missing columns: {missing}")
    geographies = geographies.dropna(subset=["geometry_wkt"]).copy()
    geographies["gadm_fullcode"] = geographies["gadm_fullcode"].astype(str)
    return geographies[required].drop_duplicates("gadm_fullcode").reset_index(drop=True)


def build_feature_collection(geographies: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    for gadm_fullcode, geometry_wkt, province_code, match_level in geographies[
        ["gadm_fullcode", "geometry_wkt", "province_code", "match_level"]
    ].itertuples(index=False, name=None):
        g = shapely.wkt.loads(geometry_wkt)
        feats.append(
            ee.Feature(
                shapely_to_ee(g),
                {
                    "gadm_fullcode": str(gadm_fullcode),
                    "province_code": int(province_code),
                    "match_level": str(match_level),
                },
            )
        )
    return ee.FeatureCollection(feats)


def define_windows() -> list[tuple[str, str]]:
    windows = []
    for start_s, end_s in BASE_WINDOWS:
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s)
        while start < end:
            next_end = min(start + pd.DateOffset(months=BATCH_MONTHS), end)
            windows.append((start.strftime("%Y-%m-%d"), next_end.strftime("%Y-%m-%d")))
            start = next_end
    return windows


def write_output(df: pd.DataFrame) -> None:
    out_path = GENERATED_DATA / "13_aod_monthly_kab.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nwrote {len(df):,} rows to {out_path}")


def main() -> None:
    init_gee()
    geographies = load_geographies()
    print(f"polygons: {len(geographies)}")
    fc = build_feature_collection(geographies)

    band = "Aerosol_Optical_Depth_Land_Ocean_Mean_Mean"
    rows_all = []
    for start, end in define_windows():
        print(f"window {start} -> {end}")
        ic = (
            ee.ImageCollection("MODIS/061/MOD08_M3").filterDate(start, end).select(band)
        )

        def reduce_one(img):
            means = img.reduceRegions(
                collection=fc, reducer=ee.Reducer.mean(), scale=111319, tileScale=4
            )
            return means.map(
                lambda f: f.set(
                    {
                        "year": img.date().get("year"),
                        "month": img.date().get("month"),
                    }
                )
            )

        flat = ic.map(reduce_one).flatten()
        t0 = time.time()
        info = flat.getInfo()
        print(
            f"  fetched {len(info['features'])} (kab × month) in {time.time() - t0:.1f}s"
        )
        for f in info["features"]:
            p = f["properties"]
            rows_all.append(
                {
                    "gadm_fullcode": str(p["gadm_fullcode"]),
                    "province_code": int(p["province_code"]),
                    "match_level": str(p["match_level"]),
                    "year": int(p["year"]),
                    "month": int(p["month"]),
                    "aod": p.get("mean"),  # default reducer output property name
                }
            )

    df = pd.DataFrame(rows_all)
    # MODIS AOD is stored scaled by 1000; convert to physical units
    df["aod"] = df.aod / 1000.0
    df = AOD_MONTHLY_SCHEMA.validate(df)
    write_output(df)
    print(df.aod.describe().round(3))
    print("\n2015 haze months (Sep-Nov) on Sumatra/Kalimantan:")
    haze = df[(df.year == 2015) & (df.month.isin([9, 10, 11]))]
    haze = haze.copy()
    haze["region"] = haze.province_code.apply(
        lambda p: (
            "Sumatra" if 11 <= p <= 21 else ("Kalimantan" if 61 <= p <= 64 else "Other")
        )
    )
    summ = haze.groupby("region").aod.describe().round(3)
    print(summ[[c for c in ["count", "mean", "50%", "max"] if c in summ.columns]])


if __name__ == "__main__":
    main()
