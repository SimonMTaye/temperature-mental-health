"""Pull ERA5-Land HOURLY temperature_2m per IFLS geography across the IFLS4 + IFLS5
fielding windows.

Same source (ERA5-Land), same polygons, same reduction as the daily pull
(03_fetch_temperature_gee.py), but hourly granularity so interview-hour heat can
be matched to each respondent (IFLS4 has interview begin/end times in bk_cov;
IFLS5 has only the date — so this is most useful for IFLS4 within-day analysis
and as a richer source for the daytime-only / nighttime-only Tmax/Tmin checks).

Approach
--------
For each batch of 2 hours, do ONE GEE reduceRegions over all deduplicated GADM
polygons -> hourly means written into a long parquet keyed
(gadm_fullcode, datetime_utc).

  BATCH_HOURS is capped at 2 because 2,211 polygons x 2 hours = 4,422 features
  per getInfo() call, just under GEE's ~5,000 feature response limit.
  To compensate, batches are fetched in parallel via ThreadPoolExecutor
  (getInfo is I/O-bound).  With 4 workers the wall-clock time is ~1/4 of serial.

Variables (ERA5-Land native ~9 km, polygon-mean):
  tmean_c_hour  hourly 2m air temperature (deg C; converted from Kelvin)
  dewp_c_hour   hourly 2m dewpoint (deg C; for hourly humidity / heat-index)

Coverage
--------
  IFLS4: ~445 days x 24h = ~10,680 hours per polygon
  IFLS5: ~505 days x 24h = ~12,120 hours per polygon
  Total: ~22,800 hours x N geographies

Runtime estimate
----------------
  Number of batches per wave: ceil((days * 24) / 2) hours = ~4,900 (IFLS4) + ~6,100 (IFLS5)
  At ~8-12s per call serial: ~25-35 hours.  With 4 workers in parallel: ~6-9 hours.
  Per-wave caches let you resume after interruption.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import ee
import pandas as pd
import shapely.wkt
from tqdm.auto import tqdm

from config import GEE_PROEJCT_ID, GENERATED_DATA, TMP_TEMPERATURE_HOURLY as TMP
from _schemas import HOURLY_TEMPERATURE_SCHEMA
from log import log

# 2 hours x 2,211 polygons = 4,422 features per getInfo() call.
# GEE's getInfo response limit is ~5,000 features, so BATCH_HOURS cannot
# be raised without first splitting geographies into separate calls.
BATCH_HOURS = 2
MAX_WORKERS = 4  # Conservative: 4 concurrent getInfo() calls, well under GEE's ~3 req/s limit

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
    rows = geographies[
        ["gadm_fullcode", "geometry_wkt", "province_code", "match_level"]
    ].itertuples(index=False, name=None)
    for gadm_fullcode, geometry_wkt, province_code, match_level in tqdm(
        rows,
        total=len(geographies),
        desc="ERA5 hourly polygons",
        unit="polygon",
        leave=False,
    ):
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
                "gadm_fullcode": str(p["gadm_fullcode"]),
                "province_code": int(p["province_code"]),
                "match_level": str(p["match_level"]),
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


def pull_window_with_retry(
    start: pd.Timestamp, end_excl: pd.Timestamp, fc: ee.FeatureCollection,
    max_retries: int = 3,
) -> pd.DataFrame:
    """pull_window with exponential-backoff retry, safe for threaded use."""
    for attempt in range(max_retries):
        try:
            return pull_window(start, end_excl, fc)
        except Exception:
            if attempt == max_retries - 1:
                raise
            wait = (attempt + 1) * 30
            log(
                f"    {start} error, retrying in {wait}s "
                f"(attempt {attempt + 1}/{max_retries})",
                "WARNING",
            )
            time.sleep(wait)


def define_windows(ind: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    interview_date = pd.to_datetime(ind.interview_datetime).dt.normalize()
    dated = ind.assign(interview_date=interview_date)
    w4 = dated[dated.wave == "IFLS4"]
    w5 = dated[dated.wave == "IFLS5"]
    return [
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


def read_cached_window(path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    cached = pd.read_parquet(path)
    if "gadm_fullcode" not in cached.columns:
        log(f"ignoring old kabupaten_code cache at {path}", "WARNING")
        return None
    return cached


def write_output(df: pd.DataFrame) -> None:
    out_path = GENERATED_DATA / "11_hourly_temperature_kab.parquet"
    df.to_parquet(out_path, index=False)
    log(f"wrote {len(df):,} rows to {out_path}")


def main() -> None:
    init_gee()
    TMP.mkdir(parents=True, exist_ok=True)

    geographies = load_geographies()
    log(f"polygons to process: {len(geographies)}")

    ind = pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")
    windows = define_windows(ind)
    log("windows:")
    for tag, a, b in windows:
        n_days = (b - a).days + 1
        n_hours = n_days * 24
        n_batches = (n_hours + BATCH_HOURS - 1) // BATCH_HOURS
        log(
            f"  {tag}: {a.date()} -> {b.date()}  "
            f"({n_days:,} days, {n_hours:,} hours, ~{n_batches:,} batches)"
        )

    fc = build_feature_collection(geographies)

    all_frames = []
    for tag, start, end in windows:
        out_path = TMP / f"{tag}_hourly_temp.parquet"
        cached = read_cached_window(out_path)
        if cached is not None:
            log(f"  {tag}: cached at {out_path}  -> skipping pull")
            all_frames.append(cached)
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
        log(
            f"  {tag}: pulling {n_batches} batches of {BATCH_HOURS}h "
            f"({BATCH_HOURS * len(geographies)} features per call)"
        )

        # --- Build batch windows ------------------------------------------------
        batch_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for s in starts:
            batch_start = pd.Timestamp(s)
            candidate_end = batch_start + timedelta(hours=BATCH_HOURS)
            e_excl = (
                candidate_end if candidate_end <= end_excl_total else end_excl_total
            )
            batch_windows.append((batch_start, e_excl))

        # --- Fetch in parallel (getInfo is I/O-bound) --------------------------
        wave_frames = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_window = {
                executor.submit(pull_window_with_retry, bs, be, fc): (bs, be)
                for bs, be in batch_windows
            }
            pbar = tqdm(
                as_completed(future_to_window),
                total=len(future_to_window),
                desc=f"{tag} ERA5 hourly",
                unit="batch",
            )
            for i, fut in enumerate(pbar, 1):
                batch_start, e_excl = future_to_window[fut]
                try:
                    df = fut.result()
                    wave_frames.append(df)
                except Exception as exc:
                    log(
                        f"    {batch_start} FAILED after all retries: {exc}",
                        "ERROR",
                    )
                    raise

                el = time.time() - t0
                eta = el / i * (len(future_to_window) - i)
                n_rows = sum(len(f) for f in wave_frames)
                pbar.set_postfix(
                    elapsed_min=f"{el / 60:.1f}",
                    eta_min=f"{eta / 60:.1f}",
                    rows=f"{n_rows:,}",
                )

        wave_df = pd.concat(wave_frames, ignore_index=True)
        wave_df.to_parquet(out_path, index=False)
        log(f"  {tag}: wrote {len(wave_df):,} rows to {out_path}")
        all_frames.append(wave_df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined["datetime_utc"] = pd.to_datetime(combined.datetime_utc, utc=True)
    combined = combined.sort_values(["gadm_fullcode", "datetime_utc"]).reset_index(
        drop=True
    )
    combined = HOURLY_TEMPERATURE_SCHEMA.validate(combined)
    write_output(combined)
    log("variable summary:", "DEBUG")
    log(combined[["tmean_c_hour", "dewp_c_hour"]].describe().round(2), "DEBUG")
    log(
        "Note: datetime_utc is in UTC. Indonesia time zones to convert to local:",
        "DEBUG",
    )
    log(
        "  WIB (UTC+7):  Sumatra, Java, West Kalimantan, Central Kalimantan -- prov codes 11-36, 61-62",
        "DEBUG",
    )
    log(
        "  WITA (UTC+8): Bali, NTB, NTT, South & East Kalimantan, Sulawesi  -- prov codes 51-53, 63-64, 71-76",
        "DEBUG",
    )
    log(
        "  WIT (UTC+9):  Maluku, Maluku Utara, Papua, Papua Barat            -- prov codes 81-82, 91-94",
        "DEBUG",
    )


if __name__ == "__main__":
    main()
