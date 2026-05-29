"""Pull MERRA-2 daily PM2.5 polygon-mean per kabupaten over IFLS4 + IFLS5 windows.

PM2.5 is constructed from MERRA-2 surface aerosol mass mixing ratios using the
standard van Donkelaar formula:

    PM2.5 (µg/m³) = (BCSMASS + 1.4 * OCSMASS + 1.375 * SO4SMASS
                     + DUSMASS25 + SSSMASS25) * 1e9

Source: NASA/GSFC/MERRA/aer/2 (hourly, ~50 km native). We aggregate the 24 hourly
images per day to a daily mean before reduceRegions over kab polygons.

Output: data/generated/pm25_daily_kab.parquet (kabupaten_code, date, pm25_ugm3, +components)
"""

import time
from datetime import timedelta

import ee
import pandas as pd
import shapely.wkt

from config import GEE_PROEJCT_ID, TMP_PM25 as TMP, GENERATED_DATA
from _schemas import PM25_DAILY_SCHEMA

WINDOWS = [
    ("2007-06-06", "2008-08-25"),  # IFLS4 (~445 days)
    ("2014-08-07", "2015-12-25"),  # IFLS5 (~505 days; covers 2015 haze peak)
]

PM25_BANDS = ["BCSMASS", "OCSMASS", "SO4SMASS", "DUSMASS25", "SSSMASS25"]


def init_gee() -> None:
    ee.Initialize(project=GEE_PROEJCT_ID)


def shapely_to_ee(g) -> ee.Geometry:
    g = g.simplify(0.05, preserve_topology=True)
    return ee.Geometry(g.__geo_interface__, opt_geodesic=False, opt_evenOdd=True)


def build_fc(kab: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    for geometry_wkt, kabupaten_code in kab[
        ["geometry_wkt", "kabupaten_code"]
    ].itertuples(index=False, name=None):
        g = shapely.wkt.loads(geometry_wkt)
        feats.append(
            ee.Feature(shapely_to_ee(g), {"kabupaten_code": int(kabupaten_code)})
        )
    return ee.FeatureCollection(feats)


def daily_pm25_image(date_str: str) -> ee.Image:
    """Mean of 24 hourly aerosol images on `date_str`, then PM2.5 formula."""
    start = ee.Date(date_str)
    end = start.advance(1, "day")
    daily = (
        ee.ImageCollection("NASA/GSFC/MERRA/aer/2")
        .filterDate(start, end)
        .select(PM25_BANDS)
        .mean()
    )
    pm25 = (
        daily.expression(
            "(BC + 1.4 * OC + 1.375 * SO4 + DU + SS) * 1e9",
            {
                "BC": daily.select("BCSMASS"),
                "OC": daily.select("OCSMASS"),
                "SO4": daily.select("SO4SMASS"),
                "DU": daily.select("DUSMASS25"),
                "SS": daily.select("SSSMASS25"),
            },
        )
        .rename("pm25_ugm3")
        .set("system:time_start", start.millis())
    )
    return pm25.addBands(daily)


def pull_window(
    start: pd.Timestamp, end_excl: pd.Timestamp, fc: ee.FeatureCollection
) -> pd.DataFrame:
    """Server-side reduceRegions across N days × all polygons."""
    dates = pd.date_range(start, end_excl - timedelta(days=1), freq="D")
    images = [daily_pm25_image(d.strftime("%Y-%m-%d")) for d in dates]
    ic = ee.ImageCollection.fromImages(images)

    def reduce_one(img):
        means = img.reduceRegions(
            collection=fc, reducer=ee.Reducer.mean(), scale=55000, tileScale=4
        )
        return means.map(lambda f: f.set("date", img.date().format("YYYY-MM-dd")))

    flat = ic.map(reduce_one).flatten()
    info = flat.getInfo()
    rows = []
    for f in info["features"]:
        p = f["properties"]
        rows.append(
            {
                "kabupaten_code": int(p["kabupaten_code"]),
                "date": p["date"],
                "pm25_ugm3": p.get("pm25_ugm3"),
                "BCSMASS": p.get("BCSMASS"),
                "OCSMASS": p.get("OCSMASS"),
                "SO4SMASS": p.get("SO4SMASS"),
                "DUSMASS25": p.get("DUSMASS25"),
                "SSSMASS25": p.get("SSSMASS25"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    init_gee()
    TMP.mkdir(parents=True, exist_ok=True)

    kab = pd.read_parquet(GENERATED_DATA / "02_kabupaten_polygons.parquet").dropna(
        subset=["geometry_wkt"]
    )
    fc = build_fc(kab)
    print(f"polygons: {len(kab)}")

    BATCH_DAYS = 12  # 303 * 12 = 3636 features per call (well under 5000 cap)
    all_frames = []
    for tag, start_s, end_s in [
        ("IFLS4", *WINDOWS[0]),
        ("IFLS5", *WINDOWS[1]),
    ]:
        cache = TMP / f"{tag}_pm25.parquet"
        if cache.exists():
            print(f"{tag}: cached")
            all_frames.append(pd.read_parquet(cache))
            continue
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s)
        end_excl_total = end + timedelta(days=1)
        if not isinstance(end_excl_total, pd.Timestamp):
            raise ValueError(f"{tag}: invalid end timestamp {end_excl_total}")
        starts = pd.date_range(start, end, freq=f"{BATCH_DAYS}D")
        wave_frames = []
        t0 = time.time()
        for i, s in enumerate(starts, 1):
            batch_start = pd.Timestamp(s)
            if not isinstance(batch_start, pd.Timestamp):
                raise ValueError(f"{tag}: invalid batch timestamp {s}")
            candidate_end = batch_start + timedelta(days=BATCH_DAYS)
            e_excl = (
                candidate_end if candidate_end <= end_excl_total else end_excl_total
            )
            try:
                df = pull_window(batch_start, e_excl, fc)
                wave_frames.append(df)
            except Exception as e:
                print(
                    f"  ERROR at {batch_start.date()}: {e}; sleeping 30s and retrying"
                )
                time.sleep(30)
                df = pull_window(batch_start, e_excl, fc)
                wave_frames.append(df)
            print(
                f"  {tag} batch {i}/{len(starts)} ({batch_start.date()} -> {e_excl.date()})  "
                f"rows={len(df)}  elapsed={time.time() - t0:.0f}s"
            )
        wave_df = pd.concat(wave_frames, ignore_index=True)
        wave_df.to_parquet(cache, index=False)
        all_frames.append(wave_df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined.date)
    combined = combined.sort_values(["kabupaten_code", "date"]).reset_index(drop=True)
    combined.to_parquet(GENERATED_DATA / "12_pm25_daily_kab.parquet", index=False)
    print(
        f"\nwrote {len(combined):,} rows to {GENERATED_DATA / '12_pm25_daily_kab.parquet'}"
    )
    print(combined.pm25_ugm3.describe().round(2))

    print("\n2015 haze months Sumatra+Kalimantan PM2.5:")
    haze = combined[
        (combined.date >= "2015-09-01") & (combined.date <= "2015-11-30")
    ].merge(kab[["kabupaten_code", "province_code"]], on="kabupaten_code", how="left")
    haze["region"] = haze.province_code.apply(
        lambda p: (
            "Sumatra" if 11 <= p <= 21 else ("Kalimantan" if 61 <= p <= 64 else "Other")
        )
    )
    print(
        haze.groupby("region")
        .pm25_ugm3.describe()
        .round(2)[["count", "mean", "50%", "max"]]
    )
    PM25_DAILY_SCHEMA.validate(combined)


if __name__ == "__main__":
    main()
