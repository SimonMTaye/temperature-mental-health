"""Pull MODIS Terra monthly Aerosol Optical Depth (AOD) per kabupaten.

AOD at 550 nm is the workhorse satellite proxy for PM2.5 / haze severity. The
2015 Indonesian peat-fire haze (driven by El Niño + drained peatland fires) shows
up as a 0.5+ AOD spike across Sumatra and Kalimantan in Sep-Nov 2015.

Source: MODIS/061/MOD08_M3, band Aerosol_Optical_Depth_Land_Ocean_Mean_Mean.

Output: data/generated/aod_monthly_kab.parquet
  cols: kabupaten_code, year, month, aod
"""

import time

import ee
import pandas as pd
import shapely.wkt

from config import GENERATED_DATA, GEE_PROEJCT_ID
from _schemas import AOD_MONTHLY_SCHEMA

# Monthly windows. Each <=12 months × 303 kab = <=3636 features per call (under 5000 limit).
WINDOWS = [
    ("2007-06-01", "2008-06-01"),  # IFLS4 part 1
    ("2008-06-01", "2008-09-01"),  # IFLS4 part 2
    ("2014-08-01", "2015-08-01"),  # IFLS5 pre-haze
    ("2015-08-01", "2016-01-01"),  # IFLS5 haze + tail
]


def init_gee() -> None:
    ee.Initialize(project=GEE_PROEJCT_ID)


def shapely_to_ee(g) -> ee.Geometry:
    g = g.simplify(
        0.05, preserve_topology=True
    )  # MODIS is ~1° anyway, simplify aggressively
    return ee.Geometry(g.__geo_interface__, opt_geodesic=False, opt_evenOdd=True)


def main() -> None:
    init_gee()
    kab = pd.read_parquet(GENERATED_DATA / "02_kabupaten_polygons.parquet").dropna(
        subset=["geometry_wkt"]
    )
    print(f"polygons: {len(kab)}")

    feats = []
    for geometry_wkt, kabupaten_code in kab[
        ["geometry_wkt", "kabupaten_code"]
    ].itertuples(index=False, name=None):
        g = shapely.wkt.loads(geometry_wkt)
        feats.append(
            ee.Feature(shapely_to_ee(g), {"kabupaten_code": int(kabupaten_code)})
        )
    fc = ee.FeatureCollection(feats)

    band = "Aerosol_Optical_Depth_Land_Ocean_Mean_Mean"
    rows_all = []
    for start, end in WINDOWS:
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
                    "kabupaten_code": int(p["kabupaten_code"]),
                    "year": int(p["year"]),
                    "month": int(p["month"]),
                    "aod": p.get("mean"),  # default reducer output property name
                }
            )

    df = pd.DataFrame(rows_all)
    # MODIS AOD is stored scaled by 1000; convert to physical units
    df["aod"] = df.aod / 1000.0
    df = AOD_MONTHLY_SCHEMA.validate(df)
    df.to_parquet(GENERATED_DATA / "13_aod_monthly_kab.parquet", index=False)
    print(
        f"\nwrote {len(df):,} rows to {GENERATED_DATA / '13_aod_monthly_kab.parquet'}"
    )
    print(df.aod.describe().round(3))
    print("\n2015 haze months (Sep-Nov) on Sumatra/Kalimantan:")
    haze = df[(df.year == 2015) & (df.month.isin([9, 10, 11]))]
    haze_kab = haze.merge(
        kab[["kabupaten_code", "province_code"]], on="kabupaten_code", how="left"
    )
    haze_kab["region"] = haze_kab.province_code.apply(
        lambda p: (
            "Sumatra" if 11 <= p <= 21 else ("Kalimantan" if 61 <= p <= 64 else "Other")
        )
    )
    summ = haze_kab.groupby("region").aod.describe().round(3)
    print(summ[[c for c in ["count", "mean", "50%", "max"] if c in summ.columns]])


if __name__ == "__main__":
    main()
