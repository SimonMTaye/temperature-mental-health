"""Pull MERRA-2 daily PM2.5 polygon-mean per IFLS geography over IFLS4 + IFLS5 windows.

PM2.5 is constructed from MERRA-2 surface aerosol mass mixing ratios using the
standard van Donkelaar formula:

    PM2.5 (µg/m³) = (BCSMASS + 1.4 * OCSMASS + 1.375 * SO4SMASS
                     + DUSMASS25 + SSSMASS25) * 1e9

Source: NASA/GSFC/MERRA/aer/2 (hourly, ~50 km native). We aggregate the 24 hourly
images per day to a daily mean before reduceRegions over kab polygons.

Output: data/generated/12_pm25_daily_kab.parquet (gadm_fullcode, date, pm25_ugm3, +components)
"""

import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import ee
import pandas as pd
import shapely.wkt
from tqdm.auto import tqdm

from config import GEE_PROEJCT_ID, TMP_PM25 as TMP, GENERATED_DATA
from _schemas import PM25_DAILY_SCHEMA
from log import log

WINDOWS = [
    ("2007-06-06", "2008-08-25"),  # IFLS4 (~445 days)
    ("2014-08-07", "2015-12-25"),  # IFLS5 (~505 days; covers 2015 haze peak)
]

PM25_BANDS = ["BCSMASS", "OCSMASS", "SO4SMASS", "DUSMASS25", "SSSMASS25"]
BATCH_DAYS = 2
MAX_WORKERS = 4
KEY_COLUMNS = ["gadm_fullcode", "date"]
GEO_FINGERPRINT_COLUMN = "geo_fingerprint"


def compute_geo_fingerprint(geographies: pd.DataFrame) -> str:
    """Return a stable hash of the geometry lookup used to build GEE features."""
    payload = (
        geographies[
            ["gadm_fullcode", "province_code", "match_level", "geometry_wkt"]
        ]
        .sort_values("gadm_fullcode")
        .astype(str)
        .to_csv(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def init_gee() -> None:
    ee.Initialize(project=GEE_PROEJCT_ID)


def shapely_to_ee(g) -> ee.Geometry:
    g = g.simplify(0.05, preserve_topology=True)
    return ee.Geometry(g.__geo_interface__, opt_geodesic=False, opt_evenOdd=True)


def load_geographies() -> pd.DataFrame:
    geographies = pd.read_parquet(GENERATED_DATA / "02_kabupaten_polygons.parquet")
    required = ["gadm_fullcode", "geometry_wkt", "province_code", "match_level"]
    missing = [col for col in required if col not in geographies.columns]
    if missing:
        raise ValueError(f"02_kabupaten_polygons.parquet missing columns: {missing}")
    geographies = geographies.dropna(subset=["geometry_wkt"]).copy()
    geographies = geographies[required].drop_duplicates("gadm_fullcode").reset_index(
        drop=True
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
                "gadm_fullcode": str(p["gadm_fullcode"]),
                "province_code": int(p["province_code"]),
                "match_level": str(p["match_level"]),
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
        if (
            not current
            or (
                len(current) < BATCH_DAYS
                and date == current[-1] + pd.Timedelta(days=1)
            )
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
    """Fetch missing PM2.5 rows using only the geographies missing for each date."""
    if missing_keys.empty:
        return pd.DataFrame()

    missing_keys = missing_keys.copy()
    missing_keys["date"] = pd.to_datetime(missing_keys.date).dt.normalize()
    missing_dates = pd.Index(sorted(missing_keys.date.unique()))
    geo_lookup = geographies.set_index("gadm_fullcode", drop=False)
    date_batches = build_missing_date_batches(missing_dates)
    log(
        f"{tag}: MERRA PM2.5 missing has {len(date_batches):,} EE calls for "
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
            executor.submit(
                pull_missing_batch, start, end_excl, batch_geographies
            ): (start, end_excl, batch_keys)
            for start, end_excl, batch_keys, batch_geographies in tasks
        }
        pbar = tqdm(
            as_completed(future_to_batch),
            total=len(future_to_batch),
            desc=f"{tag} MERRA PM2.5 missing",
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
                rows=f"{total_rows:,}",
                elapsed_s=f"{el:.0f}",
                eta_s=f"{eta:.0f}",
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
    end_excl_total = end + timedelta(days=1)
    if not isinstance(end_excl_total, pd.Timestamp):
        raise ValueError(f"{tag}: invalid end timestamp {end_excl_total}")
    starts = pd.date_range(start, end, freq=f"{BATCH_DAYS}D")
    log(
        f"{tag}: pulling {(end - start).days + 1} days x "
        f"{len(geographies)} polygons in {len(starts)} batches "
        f"with {MAX_WORKERS} workers"
    )
    batch_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for s in starts:
        batch_start = pd.Timestamp(s)
        if not isinstance(batch_start, pd.Timestamp):
            raise ValueError(f"{tag}: invalid batch timestamp {s}")
        candidate_end = batch_start + timedelta(days=BATCH_DAYS)
        e_excl = candidate_end if candidate_end <= end_excl_total else end_excl_total
        batch_windows.append((batch_start, e_excl))

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
            desc=f"{tag} MERRA PM2.5",
            unit="batch",
        )
        for i, fut in enumerate(pbar, 1):
            batch_start, e_excl = future_to_window[fut]
            try:
                df = fut.result()
            except Exception as exc:
                log(
                    f"ERROR at {batch_start.date()} after all retries: {exc}",
                    "ERROR",
                )
                raise
            wave_frames.append(df)
            total_rows += len(df)
            el = time.time() - t0
            eta = el / i * (len(future_to_window) - i)
            pbar.set_postfix(
                rows=f"{total_rows:,}",
                elapsed_s=f"{el:.0f}",
                eta_s=f"{eta:.0f}",
            )
    wave_df = pd.concat(wave_frames, ignore_index=True)
    return add_cache_metadata(wave_df, fetch_time, geo_fingerprint)


def write_output(df: pd.DataFrame) -> None:
    out_path = GENERATED_DATA / "12_pm25_daily_kab.parquet"
    df.to_parquet(out_path, index=False)
    log(f"wrote {len(df):,} rows to {out_path}")


def main() -> None:
    init_gee()
    TMP.mkdir(parents=True, exist_ok=True)

    geographies = load_geographies()
    geo_fingerprint = geographies.attrs[GEO_FINGERPRINT_COLUMN]
    log(f"polygons: {len(geographies)}")

    all_frames = []
    for tag, start_s, end_s in [
        ("IFLS4", *WINDOWS[0]),
        ("IFLS5", *WINDOWS[1]),
    ]:
        cache = TMP / f"{tag}_pm25.parquet"
        start_candidate = pd.Timestamp(start_s)
        end_candidate = pd.Timestamp(end_s)
        if not isinstance(start_candidate, pd.Timestamp):
            raise ValueError(f"{tag}: invalid start timestamp {start_s}")
        if not isinstance(end_candidate, pd.Timestamp):
            raise ValueError(f"{tag}: invalid end timestamp {end_s}")
        start = start_candidate
        end = end_candidate
        fetch_time = pd.Timestamp.now(tz="UTC")
        required_keys = build_required_keys(geographies, start, end)
        cached = read_cached_window(cache, geo_fingerprint)
        cached = None if cached is not None and cached.empty else cached
        missing_keys = missing_required_keys(required_keys, cached)
        log(
            f"{tag}: cache has {0 if cached is None else len(cached):,} rows; "
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
        wave_df.to_parquet(cache, index=False)
        all_frames.append(wave_df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop(
        columns=["fetch_time", GEO_FINGERPRINT_COLUMN], errors="ignore"
    )
    combined["date"] = pd.to_datetime(combined.date)
    combined = combined.sort_values(["gadm_fullcode", "date"]).reset_index(drop=True)
    combined = PM25_DAILY_SCHEMA.validate(combined)
    write_output(combined)
    log(combined.pm25_ugm3.describe().round(2), "DEBUG")

    log("2015 haze months Sumatra+Kalimantan PM2.5:", "DEBUG")
    haze = combined[
        (combined.date >= "2015-09-01") & (combined.date <= "2015-11-30")
    ].copy()
    haze["region"] = haze.province_code.apply(
        lambda p: (
            "Sumatra" if 11 <= p <= 21 else ("Kalimantan" if 61 <= p <= 64 else "Other")
        )
    )
    log(
        haze.groupby("region")
        .pm25_ugm3.describe()
        .round(2)[["count", "mean", "50%", "max"]],
        "DEBUG",
    )


if __name__ == "__main__":
    main()
