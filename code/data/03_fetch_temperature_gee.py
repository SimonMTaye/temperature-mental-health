"""Pull ERA5-Land daily polygon-mean temperature for every IFLS kabupaten over the
IFLS4 + IFLS5 fielding windows.

Approach
--------
For each day in the union of field windows, do ONE GEE reduceRegions over all 303
kabupaten polygons -> daily means written into a long parquet keyed (kabupaten_code, date).

Window: 30 days BEFORE earliest interview to 7 days AFTER latest. Gives us lead/lag
room for placebo and lag-effect specifications without re-pulling later.

Variables (all at ERA5-Land native ~9 km, polygon-mean):
  tmean_c       mean 2m air temperature (°C)
  tmax_c        daily max
  tmin_c        daily min (proxies nighttime min in the tropics)
  dewp_c        dewpoint at 2m
  rh_pct        derived relative humidity
  precip_mm     daily precip total
  heat_idx_c    heat index (Steadman) — co-stressor with raw heat

Output: data/generated/daily_temperature_kab.parquet  (long, ~285k rows)
"""
from __future__ import annotations

import math
import os
import time
from datetime import timedelta

import ee
import pandas as pd
import shapely.wkt

from config import GEE_ENV_PATH, OUT, TMP_TEMPERATURE as TMP
from _schemas import DAILY_TEMPERATURE_SCHEMA


def init_gee() -> None:
    for line in GEE_ENV_PATH.read_text().splitlines():
        if line.startswith("GEE_PROJECT_ID="):
            os.environ["GEE_PROJECT_ID"] = line.split("=", 1)[1].strip()
    ee.Initialize(project=os.environ["GEE_PROJECT_ID"])


def shapely_to_ee(g) -> ee.Geometry:
    # Simplify to ~0.02° (~2 km) — vastly smaller than ERA5-Land's 9 km grid, so polygon-mean
    # is unchanged. Cuts payload ~10x and lets all 303 kabs fit in one reduceRegions call.
    g = g.simplify(0.02, preserve_topology=True)
    return ee.Geometry(g.__geo_interface__, opt_geodesic=False, opt_evenOdd=True)


def build_polygon_collection(kab: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    for r in kab.itertuples(index=False):
        g = shapely.wkt.loads(r.geometry_wkt)
        feats.append(ee.Feature(shapely_to_ee(g), {"kabupaten_code": int(r.kabupaten_code)}))
    return ee.FeatureCollection(feats)


BANDS = [
    "temperature_2m",
    "temperature_2m_max",
    "temperature_2m_min",
    "dewpoint_temperature_2m",
    "total_precipitation_sum",
]


def pull_window(start: pd.Timestamp, end_excl: pd.Timestamp, fc: ee.FeatureCollection) -> pd.DataFrame:
    """One server-side reduceRegions across N days × all polygons. Keeps payload < limit."""
    ic = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start.strftime("%Y-%m-%d"), end_excl.strftime("%Y-%m-%d"))
        .select(BANDS)
    )

    def reduce_one(img):
        means = img.reduceRegions(
            collection=fc, reducer=ee.Reducer.mean(), scale=11132, tileScale=4
        )
        return means.map(lambda f: f.set("date", img.date().format("YYYY-MM-dd")))

    flat = ic.map(reduce_one).flatten()
    info = flat.getInfo()
    rows = []
    for f in info["features"]:
        p = f["properties"]
        if "temperature_2m" not in p:
            continue
        rows.append({
            "kabupaten_code": int(p["kabupaten_code"]),
            "date": p["date"],
            "tmean_c": p["temperature_2m"] - 273.15,
            "tmax_c": p["temperature_2m_max"] - 273.15,
            "tmin_c": p["temperature_2m_min"] - 273.15,
            "dewp_c": p["dewpoint_temperature_2m"] - 273.15,
            "precip_mm": (p.get("total_precipitation_sum") or 0.0) * 1000.0,
        })
    return pd.DataFrame(rows)


def derive_humidity_and_heat_index(df: pd.DataFrame) -> pd.DataFrame:
    """RH from T and Td via Magnus; heat-index via Steadman (NWS) approximation."""
    T = df.tmean_c
    Td = df.dewp_c
    a, b = 17.625, 243.04
    es = lambda x: 6.1094 * (math.e ** ((a * x) / (b + x)))
    df["rh_pct"] = 100.0 * (es(Td) / es(T))
    # Steadman heat index in Fahrenheit, then convert back
    Tf = T * 9 / 5 + 32
    R = df.rh_pct
    HI_f = (
        -42.379 + 2.04901523 * Tf + 10.14333127 * R
        - 0.22475541 * Tf * R - 6.83783e-3 * Tf**2
        - 5.481717e-2 * R**2 + 1.22874e-3 * Tf**2 * R
        + 8.5282e-4 * Tf * R**2 - 1.99e-6 * Tf**2 * R**2
    )
    HI_f = HI_f.where(Tf >= 80, Tf)  # below 80 F just use Tf
    df["heat_idx_c"] = (HI_f - 32) * 5 / 9
    return df


def main() -> None:
    init_gee()
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    kab = pd.read_parquet(OUT / "kabupaten_polygons.parquet")
    kab = kab.dropna(subset=["geometry_wkt"]).reset_index(drop=True)
    print(f"polygons to process: {len(kab)}")

    # Compute date windows
    ind = pd.read_parquet(OUT / "individuals.parquet")
    win_min = ind.interview_date.min() - timedelta(days=30)
    win_max = ind.interview_date.max() + timedelta(days=7)
    # Holes between IFLS4 and IFLS5 are wasted; pull each wave window separately
    w4 = ind[ind.wave == "IFLS4"]
    w5 = ind[ind.wave == "IFLS5"]
    windows = [
        ("IFLS4", w4.interview_date.min() - timedelta(days=30), w4.interview_date.max() + timedelta(days=7)),
        ("IFLS5", w5.interview_date.min() - timedelta(days=30), w5.interview_date.max() + timedelta(days=7)),
    ]
    print("windows:")
    for tag, a, b in windows:
        print(f"  {tag}: {a.date()} -> {b.date()}  ({(b-a).days} days)")

    fc = build_polygon_collection(kab)

    all_frames = []
    for tag, start, end in windows:
        out_path = TMP / f"{tag}_daily_temp.parquet"
        if out_path.exists():
            print(f"  {tag}: cached at {out_path}")
            all_frames.append(pd.read_parquet(out_path))
            continue
        # Batch in 15-day windows: ~4500 features/call, well under getInfo limits
        BATCH_DAYS = 15
        starts = pd.date_range(start, end, freq=f"{BATCH_DAYS}D")
        print(f"  {tag}: pulling {(end-start).days+1} days x {len(kab)} polygons in {len(starts)} batches")
        wave_frames = []
        t0 = time.time()
        for i, s in enumerate(starts, 1):
            e_excl = min(s + timedelta(days=BATCH_DAYS), end + timedelta(days=1))
            try:
                df = pull_window(s, e_excl, fc)
                wave_frames.append(df)
            except Exception as exc:
                print(f"    {s.date()}-{e_excl.date()}  ERROR: {exc}; sleeping 30s and retrying once")
                time.sleep(30)
                df = pull_window(s, e_excl, fc)
                wave_frames.append(df)
            el = time.time() - t0
            eta = el / i * (len(starts) - i)
            print(f"    {tag} batch {i}/{len(starts)}  ({s.date()} -> {e_excl.date()})  elapsed={el:.0f}s  eta={eta:.0f}s  rows={len(df)}")
        wave_df = pd.concat(wave_frames, ignore_index=True)
        wave_df.to_parquet(out_path, index=False)
        print(f"  {tag}: wrote {len(wave_df):,} rows to {out_path}")
        all_frames.append(wave_df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = derive_humidity_and_heat_index(combined)
    combined["date"] = pd.to_datetime(combined.date)
    combined = combined.sort_values(["kabupaten_code", "date"]).reset_index(drop=True)
    combined = DAILY_TEMPERATURE_SCHEMA.validate(combined)
    out_path = OUT / "daily_temperature_kab.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\nwrote {len(combined):,} rows to {out_path}")
    print("variable summary (°C / mm):")
    print(combined[["tmean_c", "tmax_c", "tmin_c", "dewp_c", "rh_pct", "precip_mm", "heat_idx_c"]].describe().round(2))


if __name__ == "__main__":
    main()
