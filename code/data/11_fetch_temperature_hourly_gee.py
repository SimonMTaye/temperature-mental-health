"""Pull ERA5-Land HOURLY temperature_2m per kabupaten across the IFLS4 + IFLS5
fielding windows.

Same source (ERA5-Land), same polygons, same reduction as the daily pull
(03_fetch_temperature_gee.py), but hourly granularity so interview-hour heat can
be matched to each respondent (IFLS4 has interview begin/end times in bk_cov;
IFLS5 has only the date — so this is most useful for IFLS4 within-day analysis
and as a richer source for the daytime-only / nighttime-only Tmax/Tmin checks).

Approach
--------
For each batch of 16 hours, do ONE GEE reduceRegions over all kabupaten polygons
-> hourly means written into a long parquet keyed (kabupaten_code, datetime_utc).

  16 hours x 303 polygons = 4,848 features per call (under the 5,000 getInfo cap).

Variables (ERA5-Land native ~9 km, polygon-mean):
  tmean_c_hour  hourly 2m air temperature (deg C; converted from Kelvin)
  dewp_c_hour   hourly 2m dewpoint (deg C; for hourly humidity / heat-index)

Coverage
--------
  IFLS4: ~445 days x 24h = ~10,680 hours per polygon
  IFLS5: ~505 days x 24h = ~12,120 hours per polygon
  Total: ~22,800 hours x 303 kabs = ~7M rows

Runtime estimate
----------------
  Number of batches per wave: ceil((days * 24) / 16) hours = ~670 (IFLS4) + 760 (IFLS5)
  At ~8-12s per call, plan ~3-5 hours for the full pull. Per-wave caches let you
  resume after interruption.

Output
------
  data/generated/hourly_temperature_kab.parquet     (combined, ~200 MB)
  data/generated/_tmp_temperature_hourly/IFLS4_hourly_temp.parquet  (cache)
  data/generated/_tmp_temperature_hourly/IFLS5_hourly_temp.parquet  (cache)

How to run
----------
  python jingyao/code/data/12_fetch_temperature_hourly_gee.py

Set your GEE project id in the .env file referenced by `init_gee()` below.
The script is restartable: per-wave caches mean an interrupted run resumes
where it left off.
"""

import time
from datetime import timedelta

import ee
import pandas as pd
import shapely.wkt

from config import GEE_PROEJCT_ID, GENERATED_DATA, TMP_TEMPERATURE_HOURLY as TMP
from _schemas import HOURLY_TEMPERATURE_SCHEMA

# Hours per server-side call. 16 * 303 polygons = 4,848 features (under 5000 cap).
BATCH_HOURS = 16

BANDS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
]


def init_gee() -> None:
    """Read GEE_PROJECT_ID from the .env file and initialize Earth Engine.
    Reuses the same .env used by 03_fetch_temperature_gee.py.
    """
    ee.Initialize(project=GEE_PROEJCT_ID)


def shapely_to_ee(g) -> ee.Geometry:
    # Simplify to ~0.02 deg (~2 km) -- vastly smaller than ERA5-Land's 9 km grid,
    # so polygon-mean is unchanged. Cuts payload ~10x.
    g = g.simplify(0.02, preserve_topology=True)
    return ee.Geometry(g.__geo_interface__, opt_geodesic=False, opt_evenOdd=True)


def build_polygon_collection(kab: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    for geometry_wkt, kabupaten_code in kab[
        ["geometry_wkt", "kabupaten_code"]
    ].itertuples(index=False, name=None):
        g = shapely.wkt.loads(geometry_wkt)
        feats.append(
            ee.Feature(shapely_to_ee(g), {"kabupaten_code": int(kabupaten_code)})
        )
    return ee.FeatureCollection(feats)


def pull_window(
    start: pd.Timestamp, end_excl: pd.Timestamp, fc: ee.FeatureCollection
) -> pd.DataFrame:
    """Server-side reduceRegions across all hourly images in [start, end_excl)."""
    ic = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(
            start.strftime("%Y-%m-%dT%H:%M:%S"), end_excl.strftime("%Y-%m-%dT%H:%M:%S")
        )
        .select(BANDS)
    )

    def reduce_one(img):
        means = img.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=11132,
            tileScale=4,
        )
        return means.map(
            lambda f: f.set("datetime", img.date().format("YYYY-MM-dd'T'HH:mm:ss"))
        )

    flat = ic.map(reduce_one).flatten()
    info = flat.getInfo()
    rows = []
    for f in info["features"]:
        p = f["properties"]
        if p.get("temperature_2m") is None:
            continue
        rows.append(
            {
                "kabupaten_code": int(p["kabupaten_code"]),
                "datetime_utc": p["datetime"],
                "tmean_c_hour": p["temperature_2m"] - 273.15,
                "dewp_c_hour": (
                    p["dewpoint_temperature_2m"] - 273.15
                    if p.get("dewpoint_temperature_2m") is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    init_gee()
    TMP.mkdir(parents=True, exist_ok=True)

    kab = pd.read_parquet(GENERATED_DATA / "02_kabupaten_polygons.parquet")
    kab = kab.dropna(subset=["geometry_wkt"]).reset_index(drop=True)
    print(f"polygons to process: {len(kab)}")

    # Same windows as the daily pull (30d lead, 7d lag).
    ind = pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")
    w4 = ind[ind.wave == "IFLS4"]
    w5 = ind[ind.wave == "IFLS5"]
    windows = [
        (
            "IFLS4",
            w4.interview_date.min() - timedelta(days=30),
            w4.interview_date.max() + timedelta(days=7),
        ),
        (
            "IFLS5",
            w5.interview_date.min() - timedelta(days=30),
            w5.interview_date.max() + timedelta(days=7),
        ),
    ]
    print("windows:")
    for tag, a, b in windows:
        n_days = (b - a).days + 1
        n_hours = n_days * 24
        n_batches = (n_hours + BATCH_HOURS - 1) // BATCH_HOURS
        print(
            f"  {tag}: {a.date()} -> {b.date()}  "
            f"({n_days:,} days, {n_hours:,} hours, ~{n_batches:,} batches)"
        )

    fc = build_polygon_collection(kab)

    all_frames = []
    for tag, start, end in windows:
        out_path = TMP / f"{tag}_hourly_temp.parquet"
        if out_path.exists():
            print(f"  {tag}: cached at {out_path}  -> skipping pull")
            all_frames.append(pd.read_parquet(out_path))
            continue

        # Snap start to top-of-hour
        start_t = pd.Timestamp(start.date()) + pd.Timedelta(hours=0)
        end_excl_total = pd.Timestamp(end.date()) + pd.Timedelta(days=1)
        if not isinstance(end_excl_total, pd.Timestamp):
            raise ValueError(f"{tag}: invalid end timestamp {end_excl_total}")
        starts = pd.date_range(
            start_t, end_excl_total, freq=f"{BATCH_HOURS}h", inclusive="left"
        )
        n_batches = len(starts)
        print(
            f"  {tag}: pulling {n_batches} batches of {BATCH_HOURS}h "
            f"({BATCH_HOURS * len(kab)} features per call)"
        )

        wave_frames = []
        t0 = time.time()
        for i, s in enumerate(starts, 1):
            batch_start = pd.Timestamp(s)
            if not isinstance(batch_start, pd.Timestamp):
                raise ValueError(f"{tag}: invalid batch timestamp {s}")
            candidate_end = batch_start + timedelta(hours=BATCH_HOURS)
            e_excl = candidate_end if candidate_end <= end_excl_total else end_excl_total
            try:
                df = pull_window(batch_start, e_excl, fc)
                wave_frames.append(df)
            except Exception as exc:
                print(f"    {batch_start} ERROR: {exc}; sleeping 30s and retrying once")
                time.sleep(30)
                df = pull_window(batch_start, e_excl, fc)
                wave_frames.append(df)

            if i % 20 == 0 or i == n_batches:
                el = time.time() - t0
                eta = el / i * (n_batches - i)
                rows = sum(len(f) for f in wave_frames)
                print(
                    f"    {tag} {i}/{n_batches}  elapsed={el / 60:.1f}min  "
                    f"eta={eta / 60:.1f}min  rows={rows:,}"
                )

        wave_df = pd.concat(wave_frames, ignore_index=True)
        wave_df.to_parquet(out_path, index=False)
        print(f"  {tag}: wrote {len(wave_df):,} rows to {out_path}")
        all_frames.append(wave_df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined["datetime_utc"] = pd.to_datetime(combined.datetime_utc, utc=True)
    combined = combined.sort_values(["kabupaten_code", "datetime_utc"]).reset_index(
        drop=True
    )
    out_path = GENERATED_DATA / "11_hourly_temperature_kab.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\nwrote {len(combined):,} rows to {out_path}")
    print("variable summary:")
    print(combined[["tmean_c_hour", "dewp_c_hour"]].describe().round(2))
    print("\nNote: datetime_utc is in UTC. Indonesia time zones to convert to local:")
    print(
        "  WIB (UTC+7):  Sumatra, Java, West Kalimantan, Central Kalimantan -- prov codes 11-36, 61-62"
    )
    print(
        "  WITA (UTC+8): Bali, NTB, NTT, South & East Kalimantan, Sulawesi  -- prov codes 51-53, 63-64, 71-76"
    )
    print(
        "  WIT (UTC+9):  Maluku, Maluku Utara, Papua, Papua Barat            -- prov codes 81-82, 91-94"
    )
    HOURLY_TEMPERATURE_SCHEMA.validate(combined)


if __name__ == "__main__":
    main()
