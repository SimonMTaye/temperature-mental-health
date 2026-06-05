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
KEY_COLUMNS = ["gadm_fullcode", "date"]


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
    return geographies[required].drop_duplicates("gadm_fullcode").reset_index(drop=True)


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


def normalize_cached_window(cached: pd.DataFrame) -> pd.DataFrame:
    """Normalize cache keys and keep fetch_time as tmp-only provenance."""
    cached = cached.copy()
    cached["date"] = pd.to_datetime(cached.date).dt.normalize()
    if "fetch_time" not in cached.columns:
        cached["fetch_time"] = pd.NaT
    else:
        cached["fetch_time"] = pd.to_datetime(cached.fetch_time, utc=True)
    return cached.drop_duplicates(KEY_COLUMNS, keep="first").reset_index(drop=True)


def read_cached_window(path) -> pd.DataFrame | None:
    """Read a wave tmp cache, ignoring legacy caches keyed by kabupaten_code."""
    if not path.exists():
        return None
    cached = pd.read_parquet(path)
    if "gadm_fullcode" not in cached.columns:
        log(f"ignoring old kabupaten_code cache at {path}", "WARNING")
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


def fetch_missing_rows(
    tag: str,
    missing_keys: pd.DataFrame,
    geographies: pd.DataFrame,
    fetch_time: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch missing PM2.5 rows using only the geographies missing for each date."""
    if missing_keys.empty:
        return pd.DataFrame()

    missing_keys = missing_keys.copy()
    missing_keys["date"] = pd.to_datetime(missing_keys.date).dt.normalize()
    missing_dates = pd.Index(sorted(missing_keys.date.unique()))
    geo_lookup = geographies.set_index("gadm_fullcode", drop=False)
    frames = []
    t0 = time.time()
    n_batches = (len(missing_dates) + BATCH_DAYS - 1) // BATCH_DAYS
    batches = tqdm(
        range(0, len(missing_dates), BATCH_DAYS),
        total=n_batches,
        desc=f"{tag} MERRA PM2.5 missing",
        unit="batch",
    )
    for offset in batches:
        batch_dates = missing_dates[offset : offset + BATCH_DAYS]
        batch_keys = missing_keys[missing_keys.date.isin(batch_dates)]
        pulled = []
        for date, date_keys in batch_keys.groupby("date", sort=True):
            date_geographies = geo_lookup.loc[
                date_keys.gadm_fullcode.unique()
            ].reset_index(drop=True)
            fc = build_feature_collection(date_geographies)
            day_candidate = pd.Timestamp(date)
            if not isinstance(day_candidate, pd.Timestamp):
                raise ValueError(f"{tag}: invalid missing date {date}")
            day = day_candidate
            df = pull_window(day, day + timedelta(days=1), fc)
            pulled.append(keep_missing_rows_only(df, date_keys))
        df = pd.concat(pulled, ignore_index=True) if pulled else pd.DataFrame()
        if not df.empty:
            df["fetch_time"] = fetch_time
            frames.append(df)
        batches.set_postfix(
            rows=sum(len(frame) for frame in frames),
            elapsed_s=f"{time.time() - t0:.0f}",
        )
    return pd.concat(frames, ignore_index=True)


def pull_wave_from_scratch(
    tag: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    geographies: pd.DataFrame,
    fetch_time: pd.Timestamp,
) -> pd.DataFrame:
    """Run the original full-wave pull when no usable tmp cache exists."""
    fc = build_feature_collection(geographies)
    end_excl_total = end + timedelta(days=1)
    if not isinstance(end_excl_total, pd.Timestamp):
        raise ValueError(f"{tag}: invalid end timestamp {end_excl_total}")
    starts = pd.date_range(start, end, freq=f"{BATCH_DAYS}D")
    wave_frames = []
    t0 = time.time()
    batches = tqdm(starts, desc=f"{tag} MERRA PM2.5", unit="batch")
    for s in batches:
        batch_start = pd.Timestamp(s)
        if not isinstance(batch_start, pd.Timestamp):
            raise ValueError(f"{tag}: invalid batch timestamp {s}")
        candidate_end = batch_start + timedelta(days=BATCH_DAYS)
        e_excl = candidate_end if candidate_end <= end_excl_total else end_excl_total
        try:
            df = pull_window(batch_start, e_excl, fc)
            wave_frames.append(df)
        except Exception as e:
            log(
                f"ERROR at {batch_start.date()}: {e}; sleeping 30s and retrying",
                "WARNING",
            )
            time.sleep(30)
            df = pull_window(batch_start, e_excl, fc)
            wave_frames.append(df)
        batches.set_postfix(
            rows=len(df),
            elapsed_s=f"{time.time() - t0:.0f}",
            window=f"{batch_start.date()}->{e_excl.date()}",
        )
    wave_df = pd.concat(wave_frames, ignore_index=True)
    wave_df["fetch_time"] = fetch_time
    return wave_df


def write_output(df: pd.DataFrame) -> None:
    out_path = GENERATED_DATA / "12_pm25_daily_kab.parquet"
    df.to_parquet(out_path, index=False)
    log(f"wrote {len(df):,} rows to {out_path}")


def main() -> None:
    init_gee()
    TMP.mkdir(parents=True, exist_ok=True)

    geographies = load_geographies()
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
        cached = read_cached_window(cache)
        cached = None if cached is not None and cached.empty else cached
        missing_keys = missing_required_keys(required_keys, cached)
        log(
            f"{tag}: cache has {0 if cached is None else len(cached):,} rows; "
            f"{len(missing_keys):,} of {len(required_keys):,} keys need fetching"
        )
        if cached is None:
            wave_df = pull_wave_from_scratch(tag, start, end, geographies, fetch_time)
        else:
            fetched = fetch_missing_rows(tag, missing_keys, geographies, fetch_time)
            wave_df = normalize_cached_window(
                pd.concat([cached, fetched], ignore_index=True)
            )
        wave_df.to_parquet(cache, index=False)
        all_frames.append(wave_df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop(columns=["fetch_time"], errors="ignore")
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
