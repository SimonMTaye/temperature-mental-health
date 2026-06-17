"""Pull ERA5-Land daily polygon-mean temperature for every IFLS geography over the
IFLS4 + IFLS5 fielding windows.

Approach
--------
For each day in the union of field windows, do ONE GEE reduceRegions over all
deduplicated GADM polygons -> daily means written into a long parquet keyed
(gadm_fullcode, date).

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

Output: data/generated/10_daily_temperature_kab.parquet
"""

import math
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import ee
import pandas as pd
import shapely.wkt
from tqdm.auto import tqdm

from data.config import GEE_PROEJCT_ID, GENERATED_DATA, TMP_TEMPERATURE as TMP
from data._schemas import DAILY_TEMPERATURE_HEAT_SCHEMA
from data.log import log


def init_gee() -> None:
    ee.Initialize(project=GEE_PROEJCT_ID)


def shapely_to_ee(g) -> ee.Geometry:
    # Simplify to ~0.02° (~2 km) — vastly smaller than ERA5-Land's 9 km grid, so polygon-mean
    # is unchanged. Cuts payload ~10x and lets all 303 kabs fit in one reduceRegions call.
    g = g.simplify(0.02, preserve_topology=True)
    return ee.Geometry(g.__geo_interface__, opt_geodesic=False, opt_evenOdd=True)


def load_geographies() -> pd.DataFrame:
    geographies = pd.read_parquet(GENERATED_DATA / "02_kabupaten_polygons.parquet")
    required = ["gadm_fullcode", "geometry_wkt", "province_code", "match_level"]
    missing = [col for col in required if col not in geographies.columns]
    if missing:
        raise ValueError(f"02_kabupaten_polygons.parquet missing columns: {missing}")
    geographies = geographies.dropna(subset=["geometry_wkt"]).copy()
    geographies = (
        geographies[required].drop_duplicates("gadm_fullcode").reset_index(drop=True)
    )
    geographies.attrs[GEO_FINGERPRINT_COLUMN] = compute_geo_fingerprint(geographies)
    return geographies


def build_feature_collection(geographies: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    rows = geographies[
        ["gadm_fullcode", "geometry_wkt", "province_code", "match_level"]
    ].itertuples(index=False, name=None)
    for gadm_fullcode, geometry_wkt, province_code, match_level in rows:
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


def build_buffered_feature_collection(
    geographies: pd.DataFrame, buffer_degrees: float
) -> ee.FeatureCollection:
    feats = []
    for gadm_fullcode, geometry_wkt, province_code, match_level in geographies[
        ["gadm_fullcode", "geometry_wkt", "province_code", "match_level"]
    ].itertuples(index=False, name=None):
        g = shapely.wkt.loads(geometry_wkt).buffer(buffer_degrees)
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


BANDS = [
    "temperature_2m",
    "temperature_2m_max",
    "temperature_2m_min",
    "dewpoint_temperature_2m",
    "total_precipitation_sum",
]
BATCH_DAYS = 2
MAX_WORKERS = 4
KEY_COLUMNS = ["gadm_fullcode", "date"]
GEO_FINGERPRINT_COLUMN = "geo_fingerprint"


def compute_geo_fingerprint(geographies: pd.DataFrame) -> str:
    """Return a stable hash of the geometry lookup used to build GEE features."""
    payload = (
        geographies[["gadm_fullcode", "province_code", "match_level", "geometry_wkt"]]
        .sort_values("gadm_fullcode")
        .astype(str)
        .to_csv(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pull_window(
    start: pd.Timestamp, end_excl: pd.Timestamp, fc: ee.FeatureCollection
) -> pd.DataFrame:
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
        rows.append(
            {
                "gadm_fullcode": str(p["gadm_fullcode"]),
                "province_code": int(p["province_code"]),
                "match_level": str(p["match_level"]),
                "date": p["date"],
                "tmean_c": p["temperature_2m"] - 273.15,
                "tmax_c": p["temperature_2m_max"] - 273.15,
                "tmin_c": p["temperature_2m_min"] - 273.15,
                "dewp_c": p["dewpoint_temperature_2m"] - 273.15,
                "precip_mm": (p.get("total_precipitation_sum") or 0.0) * 1000.0,
            }
        )
    return pd.DataFrame(rows)


def pull_window_with_retry(
    start: pd.Timestamp,
    end_excl: pd.Timestamp,
    fc: ee.FeatureCollection,
    max_retries: int = 3,
) -> pd.DataFrame:
    """pull_window with exponential-backoff retry, safe for threaded use."""
    for attempt in range(max_retries):
        try:
            return pull_window(start, end_excl, fc)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = (attempt + 1) * 30
            log(
                f"    {start} error ({exc}); retrying in {wait}s "
                f"(attempt {attempt + 1}/{max_retries})",
                "WARNING",
            )
            time.sleep(wait)
    raise RuntimeError(f"failed to pull {start} after {max_retries} attempts")


def add_cache_metadata(
    df: pd.DataFrame, fetch_time: pd.Timestamp, geo_fingerprint: str
) -> pd.DataFrame:
    """Attach tmp-cache provenance columns."""
    if df.empty:
        return df
    out = df.copy()
    out["fetch_time"] = fetch_time
    out[GEO_FINGERPRINT_COLUMN] = geo_fingerprint
    return out


def derive_humidity_and_heat_index(df: pd.DataFrame) -> pd.DataFrame:
    """RH from T and Td via Magnus; heat-index via Steadman (NWS) approximation."""
    T = df.tmean_c
    Td = df.dewp_c
    a, b = 17.625, 243.04

    def es(x):
        return 6.1094 * (math.e ** ((a * x) / (b + x)))

    df["rh_pct"] = 100.0 * (es(Td) / es(T))
    # Steadman heat index in Fahrenheit, then convert back
    Tf = T * 9 / 5 + 32
    R = df.rh_pct
    HI_f = (
        -42.379
        + 2.04901523 * Tf
        + 10.14333127 * R
        - 0.22475541 * Tf * R
        - 6.83783e-3 * Tf**2
        - 5.481717e-2 * R**2
        + 1.22874e-3 * Tf**2 * R
        + 8.5282e-4 * Tf * R**2
        - 1.99e-6 * Tf**2 * R**2
    )
    HI_f = HI_f.where(Tf >= 80, Tf)  # below 80 F just use Tf
    df["heat_idx_c"] = (HI_f - 32) * 5 / 9
    return df


def add_heat_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily threshold, rolling hot-day, CDD, and local-extreme heat features."""
    df = df.sort_values(["gadm_fullcode", "date"]).reset_index(drop=True).copy()
    g = df.groupby("gadm_fullcode", group_keys=False)

    for threshold in [28, 30, 32]:
        df[f"hot{threshold}"] = (df.tmax_c >= threshold).astype(int)

    for threshold in [28, 30, 32]:
        for window in [7, 30]:
            col = f"hot{threshold}_{window}d"
            df[col] = (
                g[f"hot{threshold}"]
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )

    df["cdd"] = (df.tmean_c - 22).clip(lower=0)
    df["cdd_7d"] = (
        g["cdd"].rolling(window=7, min_periods=1).sum().reset_index(level=0, drop=True)
    )

    p90 = g["tmean_c"].transform(lambda s: s.quantile(0.90))
    df["tmean_p90_30d"] = (
        g["tmean_c"]
        .rolling(window=30, min_periods=15)
        .quantile(0.90)
        .reset_index(level=0, drop=True)
    )
    df["is_extreme"] = (df.tmean_c >= p90).astype(int)
    return df


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


def normalize_cached_window(cached: pd.DataFrame) -> pd.DataFrame:
    """Normalize cache keys and keep fetch_time as tmp-only provenance."""
    cached = cached.copy()
    cached["date"] = pd.to_datetime(cached.date).dt.normalize()
    if "fetch_time" not in cached.columns:
        cached["fetch_time"] = pd.NaT
    else:
        cached["fetch_time"] = pd.to_datetime(cached.fetch_time, utc=True)
    return cached.drop_duplicates(KEY_COLUMNS, keep="first").reset_index(drop=True)


def read_cached_window(path, geo_fingerprint: str) -> pd.DataFrame | None:
    """Read a wave tmp cache, ignoring legacy or stale-geometry caches."""
    if not path.exists():
        return None
    cached = pd.read_parquet(path)
    if "gadm_fullcode" not in cached.columns:
        log(f"ignoring old kabupaten_code cache at {path}", "WARNING")
        return None
    if GEO_FINGERPRINT_COLUMN not in cached.columns:
        log(f"ignoring cache without geography fingerprint at {path}", "WARNING")
        return None
    fingerprints = set(cached[GEO_FINGERPRINT_COLUMN].astype(str).unique())
    if fingerprints != {geo_fingerprint}:
        log(f"ignoring stale geography cache at {path}", "WARNING")
        return None
    return normalize_cached_window(cached)


def build_required_keys(
    geographies: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.MultiIndex:
    """Build the full gadm_fullcode-date key set required for one wave."""
    dates = pd.DatetimeIndex(pd.date_range(start, end, freq="D"))
    return pd.MultiIndex.from_product(
        [geographies.gadm_fullcode.unique(), dates],
        names=KEY_COLUMNS,
    )


def missing_required_keys(
    required_keys: pd.MultiIndex, cached: pd.DataFrame | None
) -> pd.DataFrame:
    """Return required keys that are absent from the existing tmp cache."""
    if cached is None:
        return required_keys.to_frame(index=False)
    cached_keys = cached[KEY_COLUMNS].copy()
    cached_keys["date"] = pd.to_datetime(cached_keys.date).dt.normalize()
    missing = required_keys.difference(pd.MultiIndex.from_frame(cached_keys))
    return missing.to_frame(index=False)


def keep_missing_rows_only(
    df: pd.DataFrame, missing_keys: pd.DataFrame
) -> pd.DataFrame:
    """Keep only rows returned by GEE that correspond to requested missing keys."""
    if df.empty:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out.date).dt.normalize()
    missing_index = pd.MultiIndex.from_frame(missing_keys[KEY_COLUMNS])
    out_index = pd.MultiIndex.from_frame(out[KEY_COLUMNS])
    return out.loc[out_index.isin(missing_index)].reset_index(drop=True)


def build_missing_date_batches(missing_dates: pd.Index) -> list[list[pd.Timestamp]]:
    """Group missing dates into consecutive batches capped at BATCH_DAYS."""
    batches: list[list[pd.Timestamp]] = []
    current: list[pd.Timestamp] = []
    for date in sorted(pd.Timestamp(date) for date in missing_dates):
        if not current or (
            len(current) < BATCH_DAYS and date == current[-1] + pd.Timedelta(days=1)
        ):
            current.append(date)
            continue
        batches.append(current)
        current = [date]
    if current:
        batches.append(current)
    return batches


def pull_missing_batch(
    start: pd.Timestamp,
    end_excl: pd.Timestamp,
    batch_geographies: pd.DataFrame,
) -> pd.DataFrame:
    fc = build_feature_collection(batch_geographies)
    return pull_window_with_retry(start, end_excl, fc)


def fetch_missing_rows(
    tag: str,
    missing_keys: pd.DataFrame,
    geographies: pd.DataFrame,
    fetch_time: pd.Timestamp,
    geo_fingerprint: str,
) -> pd.DataFrame:
    """Fetch missing daily rows using only the geographies missing for each date."""
    if missing_keys.empty:
        return pd.DataFrame()

    missing_keys = missing_keys.copy()
    missing_keys["date"] = pd.to_datetime(missing_keys.date).dt.normalize()
    missing_dates = pd.Index(sorted(missing_keys.date.unique()))
    geo_lookup = geographies.set_index("gadm_fullcode", drop=False)
    date_batches = build_missing_date_batches(missing_dates)
    log(
        f"  {tag}: ERA5 daily missing has {len(date_batches):,} EE calls for "
        f"{len(missing_keys):,} keys"
    )
    tasks = []
    for batch_dates in date_batches:
        batch_keys = missing_keys[missing_keys.date.isin(batch_dates)]
        batch_geographies = geo_lookup.loc[
            batch_keys.gadm_fullcode.unique()
        ].reset_index(drop=True)
        start = batch_dates[0]
        end_excl = batch_dates[-1] + pd.Timedelta(days=1)
        tasks.append((start, end_excl, batch_keys, batch_geographies))

    frames = []
    t0 = time.time()
    total_rows = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {
            executor.submit(pull_missing_batch, start, end_excl, batch_geographies): (
                start,
                end_excl,
                batch_keys,
            )
            for start, end_excl, batch_keys, batch_geographies in tasks
        }
        pbar = tqdm(
            as_completed(future_to_batch),
            total=len(future_to_batch),
            desc=f"{tag} ERA5 daily missing",
            unit="batch",
        )
        for i, fut in enumerate(pbar, 1):
            batch_start, e_excl, batch_keys = future_to_batch[fut]
            try:
                df = keep_missing_rows_only(fut.result(), batch_keys)
            except Exception as exc:
                log(
                    f"    {batch_start.date()}-{e_excl.date()} missing batch FAILED "
                    f"after all retries: {exc}",
                    "ERROR",
                )
                raise
            if not df.empty:
                df = add_cache_metadata(df, fetch_time, geo_fingerprint)
                frames.append(df)
                total_rows += len(df)
            el = time.time() - t0
            eta = el / i * (len(future_to_batch) - i)
            pbar.set_postfix(
                elapsed_s=f"{el:.0f}",
                eta_s=f"{eta:.0f}",
                rows=f"{total_rows:,}",
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pull_wave_from_scratch(
    tag: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    geographies: pd.DataFrame,
    fetch_time: pd.Timestamp,
    geo_fingerprint: str,
) -> pd.DataFrame:
    """Run the original full-wave pull when no usable tmp cache exists."""
    fc = build_feature_collection(geographies)
    starts = pd.date_range(start, end, freq=f"{BATCH_DAYS}D")
    log(
        f"  {tag}: pulling {(end - start).days + 1} days x "
        f"{len(geographies)} polygons in {len(starts)} batches "
        f"with {MAX_WORKERS} workers"
    )
    batch_windows = [
        (
            pd.Timestamp(s),
            min(pd.Timestamp(s) + timedelta(days=BATCH_DAYS), end + timedelta(days=1)),
        )
        for s in starts
    ]

    wave_frames = []
    t0 = time.time()
    total_rows = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_window = {
            executor.submit(pull_window_with_retry, batch_start, e_excl, fc): (
                batch_start,
                e_excl,
            )
            for batch_start, e_excl in batch_windows
        }
        pbar = tqdm(
            as_completed(future_to_window),
            total=len(future_to_window),
            desc=f"{tag} ERA5 daily",
            unit="batch",
        )
        for i, fut in enumerate(pbar, 1):
            batch_start, e_excl = future_to_window[fut]
            try:
                df = fut.result()
            except Exception as exc:
                log(
                    f"    {batch_start.date()}-{e_excl.date()} FAILED after all retries: {exc}",
                    "ERROR",
                )
                raise
            wave_frames.append(df)
            total_rows += len(df)
            el = time.time() - t0
            eta = el / i * (len(future_to_window) - i)
            pbar.set_postfix(
                elapsed_s=f"{el:.0f}",
                eta_s=f"{eta:.0f}",
                rows=f"{total_rows:,}",
            )
    wave_df = pd.concat(wave_frames, ignore_index=True)
    return add_cache_metadata(wave_df, fetch_time, geo_fingerprint)


def write_output(df: pd.DataFrame) -> None:
    out_path = GENERATED_DATA / "10_daily_temperature_kab.parquet"
    df.to_parquet(out_path, index=False)
    log(f"wrote {len(df):,} rows to {out_path}")


def repair_missing_geographies(
    geographies: pd.DataFrame,
    weather: pd.DataFrame,
    windows: list[tuple[str, pd.Timestamp, pd.Timestamp]],
    fetch_time: pd.Timestamp,
    geo_fingerprint: str,
) -> pd.DataFrame:
    """Retry geographies with no ERA5 rows using progressively buffered polygons."""
    missing_codes = sorted(set(geographies.gadm_fullcode) - set(weather.gadm_fullcode))
    if not missing_codes:
        return weather

    log("repairing small geographies with buffered ERA5 polygons", "WARNING")
    repaired = weather.copy()
    for buffer_degrees in tqdm(
        [0.03, 0.05, 0.10, 0.15],
        desc="ERA5 repair buffers",
        unit="buffer",
    ):
        missing_geographies = geographies[
            geographies.gadm_fullcode.isin(missing_codes)
        ].copy()
        log(
            f"  buffer={buffer_degrees:.2f} degrees for {len(missing_geographies)} geographies"
        )
        fc = build_buffered_feature_collection(
            missing_geographies, buffer_degrees=buffer_degrees
        )

        repair_frames = []
        for tag, start, end in windows:
            BATCH_DAYS = 2
            starts = pd.date_range(start, end, freq=f"{BATCH_DAYS}D")
            batches = tqdm(
                starts,
                desc=f"{tag} ERA5 repair {buffer_degrees:.2f}",
                unit="batch",
                leave=False,
            )
            for s in batches:
                e_excl = min(s + timedelta(days=BATCH_DAYS), end + timedelta(days=1))
                df = pull_window_with_retry(s, e_excl, fc)
                df = add_cache_metadata(df, fetch_time, geo_fingerprint)
                repair_frames.append(df)
                batches.set_postfix(rows=len(df), window=f"{s.date()}->{e_excl.date()}")

        if repair_frames:
            repaired = pd.concat([repaired, *repair_frames], ignore_index=True)
            repaired = repaired.drop_duplicates(["gadm_fullcode", "date"], keep="first")
        missing_codes = sorted(
            set(geographies.gadm_fullcode) - set(repaired.gadm_fullcode)
        )
        if not missing_codes:
            return repaired

    repaired = repaired.drop_duplicates(["gadm_fullcode", "date"], keep="first")
    if missing_codes:
        log(f"{len(missing_codes)} geographies still missing ERA5 rows", "WARNING")
    return repaired


def write_repaired_window_caches(
    weather: pd.DataFrame, windows: list[tuple[str, pd.Timestamp, pd.Timestamp]]
) -> None:
    """Write repaired daily rows back to per-wave tmp caches for future resumes."""
    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather.date).dt.normalize()
    for tag, start, end in windows:
        out_path = TMP / f"{tag}_daily_temp.parquet"
        wave = weather[(weather.date >= start) & (weather.date <= end)].copy()
        wave = normalize_cached_window(wave)
        wave.to_parquet(out_path, index=False)
        log(f"  {tag}: refreshed repaired cache with {len(wave):,} rows")


def main() -> None:
    init_gee()
    TMP.mkdir(parents=True, exist_ok=True)

    geographies = load_geographies()
    geo_fingerprint = geographies.attrs[GEO_FINGERPRINT_COLUMN]
    log(f"polygons to process: {len(geographies)}")

    ind = pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")
    windows = define_windows(ind)
    log("windows:")
    for tag, a, b in windows:
        log(f"  {tag}: {a.date()} -> {b.date()}  ({(b - a).days} days)")

    all_frames = []
    for tag, start, end in windows:
        out_path = TMP / f"{tag}_daily_temp.parquet"
        fetch_time = pd.Timestamp.now(tz="UTC")
        required_keys = build_required_keys(geographies, start, end)
        cached = read_cached_window(out_path, geo_fingerprint)
        cached = None if cached is not None and cached.empty else cached
        missing_keys = missing_required_keys(required_keys, cached)
        log(
            f"  {tag}: cache has {0 if cached is None else len(cached):,} rows; "
            f"{len(missing_keys):,} of {len(required_keys):,} keys need fetching"
        )
        if cached is None:
            wave_df = pull_wave_from_scratch(
                tag, start, end, geographies, fetch_time, geo_fingerprint
            )
        else:
            fetched = fetch_missing_rows(
                tag, missing_keys, geographies, fetch_time, geo_fingerprint
            )
            wave_df = normalize_cached_window(
                pd.concat([cached, fetched], ignore_index=True)
            )
        wave_df.to_parquet(out_path, index=False)
        log(f"  {tag}: wrote {len(wave_df):,} rows to {out_path}")
        all_frames.append(wave_df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = repair_missing_geographies(
        geographies,
        combined,
        windows,
        pd.Timestamp.now(tz="UTC"),
        geo_fingerprint,
    )
    write_repaired_window_caches(combined, windows)
    combined = combined.drop(
        columns=["fetch_time", GEO_FINGERPRINT_COLUMN], errors="ignore"
    )
    combined = derive_humidity_and_heat_index(combined)
    combined["date"] = pd.to_datetime(combined.date)
    combined = add_heat_features(combined)
    combined = DAILY_TEMPERATURE_HEAT_SCHEMA.validate(combined)

    write_output(combined)
    log("variable summary (C / mm):", "DEBUG")
    log(
        combined[
            [
                "tmean_c",
                "tmax_c",
                "tmin_c",
                "dewp_c",
                "rh_pct",
                "precip_mm",
                "heat_idx_c",
                "hot30_7d",
                "hot32_30d",
                "cdd_7d",
                "is_extreme",
            ]
        ]
        .describe()
        .round(2),
        "DEBUG",
    )


if __name__ == "__main__":
    main()
