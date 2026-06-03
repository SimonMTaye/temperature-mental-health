"""Build person-wave processed temperature variables for analysis tables.

Output: data/generated/26_processed_temperature_data.parquet
Row level: one person-wave record, keyed by pidlink + wave.
"""

import numpy as np
import pandas as pd

from _schemas import PROCESSED_TEMPERATURE_SCHEMA
from config import GENERATED_DATA
from log import log


POST_SUBSIDY_DATE = pd.Timestamp("2014-11-18")


def _utc_offset_hours(province_code: int) -> int:
    if (11 <= province_code <= 18) or (21 <= province_code <= 36) or province_code in (61, 62):
        return 7
    if (51 <= province_code <= 53) or (63 <= province_code <= 65) or (71 <= province_code <= 76):
        return 8
    if (81 <= province_code <= 82) or (91 <= province_code <= 94):
        return 9
    return 7


def _rolling_mean_excluding_today(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    return s.rolling(window, min_periods=min_periods).mean().shift(1)


def add_daily_features(temp: pd.DataFrame) -> pd.DataFrame:
    """Add daily lag, lead, anomaly, and inclusive past-week features."""
    temp = temp.sort_values(["gadm_fullcode", "date"], kind="stable").copy()
    grouped = temp.groupby("gadm_fullcode", group_keys=False)

    temp["tmean_lag1"] = grouped["tmean_c"].shift(1)
    temp["tmean_lag3"] = grouped["tmean_c"].transform(
        lambda s: _rolling_mean_excluding_today(s, 3, 1)
    )
    temp["tmean_lag7"] = grouped["tmean_c"].transform(
        lambda s: _rolling_mean_excluding_today(s, 7, 1)
    )
    temp["tmin_lag1"] = grouped["tmin_c"].shift(1)
    temp["tmax_lag1"] = grouped["tmax_c"].shift(1)
    temp["heat_idx_lag1"] = grouped["heat_idx_c"].shift(1)
    temp["tmean_base30"] = grouped["tmean_c"].transform(
        lambda s: _rolling_mean_excluding_today(s, 30, 15)
    )
    temp["tmean_lead7"] = grouped["tmean_c"].shift(-7)

    temp["tmean_7d"] = grouped["tmean_c"].transform(
        lambda s: s.rolling(7, min_periods=4).mean()
    )
    temp["hot30_7d"] = (
        temp["tmax_c"].gt(30.0)
        .astype(np.int8)
        .groupby(temp["gadm_fullcode"])
        .transform(lambda s: s.rolling(7, min_periods=4).sum())
    )
    p90 = temp.groupby("gadm_fullcode")["tmean_c"].transform(lambda s: s.quantile(0.90))
    temp["heatwave_7d"] = (
        temp["tmean_c"].gt(p90)
        .astype(np.int8)
        .groupby(temp["gadm_fullcode"])
        .transform(lambda s: s.rolling(7, min_periods=4).sum())
    )
    return temp


def merge_daily(ind: pd.DataFrame, temp: pd.DataFrame) -> pd.DataFrame:
    ind = ind.copy()
    ind["interview_date"] = pd.to_datetime(ind.interview_datetime).dt.normalize()
    keep = [
        "gadm_fullcode",
        "date",
        "match_level",
        "tmean_c",
        "tmax_c",
        "tmin_c",
        "heat_idx_c",
        "rh_pct",
        "precip_mm",
        "tmean_lag1",
        "tmean_lag3",
        "tmean_lag7",
        "tmin_lag1",
        "tmax_lag1",
        "heat_idx_lag1",
        "tmean_base30",
        "tmean_lead7",
        "tmean_7d",
        "hot30_7d",
        "heatwave_7d",
    ]
    out = ind.merge(
        temp[keep],
        left_on=["gadm_fullcode", "interview_date"],
        right_on=["gadm_fullcode", "date"],
        how="left",
        validate="m:1",
    ).drop(columns=["date"])
    out["precip_mm"] = out.precip_mm.clip(lower=0)
    out["t_anom_today"] = out.tmean_c - out.tmean_base30
    out["t_anom_lag1"] = out.tmean_lag1 - out.tmean_base30
    out["heat_bin"] = pd.cut(
        out.tmean_c,
        bins=[-np.inf, 22, 24, 26, 28, np.inf],
        labels=["<22", "22-24", "24-26", "26-28", "28+"],
    )
    for col in ["tmean_c", "tmax_c", "tmin_c", "tmean_7d", "hot30_7d", "heatwave_7d"]:
        out[f"{col}_dev"] = out[col] - out[col].mean()
    out["heat_c_dev"] = out["tmean_c_dev"]
    out["cdd_tmax30"] = (out.tmax_c - 30.0).clip(lower=0)
    out["cdd_tmax32"] = (out.tmax_c - 32.0).clip(lower=0)
    out["cdd_tmin23"] = (out.tmin_c - 23.0).clip(lower=0)
    out["cdd_tmin24"] = (out.tmin_c - 24.0).clip(lower=0)
    out["day_id"] = (
        out.interview_date.dt.year * 10000
        + out.interview_date.dt.month * 100
        + out.interview_date.dt.day
    ).astype(int)
    return out


def add_hourly_temperature(df: pd.DataFrame) -> pd.DataFrame:
    hourly_path = GENERATED_DATA / "11_hourly_temperature_kab.parquet"
    df = df.copy()
    assert hourly_path.exists(), f"Missing hourly temperature file: {hourly_path}"

    hourly = pd.read_parquet(hourly_path)
    has_gadm_key = {"gadm_fullcode", "datetime_utc", "tmean_c_hour"}.issubset(
        hourly.columns
    )
    has_legacy_kab_key = {"kabupaten_code", "datetime_utc", "tmean_c_hour"}.issubset(
        hourly.columns
    )
    assert has_gadm_key or has_legacy_kab_key, (
        "11_hourly_temperature_kab.parquet must include either gadm_fullcode or "
        "kabupaten_code, plus datetime_utc and tmean_c_hour"
    )
    hourly = hourly[
        [
            col
            for col in ["gadm_fullcode", "kabupaten_code", "datetime_utc", "tmean_c_hour"]
            if col in hourly.columns
        ]
    ]
    hourly["datetime_utc"] = pd.to_datetime(hourly.datetime_utc, utc=True)
    # ERA5 hourly timestamps are UTC; IFLS interview hours are local Indonesian time.
    df["utc_offset"] = df.province_code.map(_utc_offset_hours)
    local_hour = df.hour_start.round().clip(lower=0, upper=23).astype(int)
    df["interview_dt_utc"] = (
        df["interview_date"].dt.tz_localize("UTC")
        + pd.to_timedelta(local_hour - df["utc_offset"], unit="h")
    )
    if "gadm_fullcode" in hourly.columns:
        df = df.merge(
            hourly,
            left_on=["gadm_fullcode", "interview_dt_utc"],
            right_on=["gadm_fullcode", "datetime_utc"],
            how="left",
            validate="m:1",
        )
    else:
        df = df.merge(
            hourly,
            left_on=["kabupaten_code", "interview_dt_utc"],
            right_on=["kabupaten_code", "datetime_utc"],
            how="left",
            validate="m:1",
        )
    df = df.drop(columns=["datetime_utc", "interview_dt_utc", "utc_offset"])
    matched = df.tmean_c_hour.notna().sum()
    log(f"matched hourly temperature for {matched:,} of {len(df):,} person-wave rows")
    df["heat_hr_dev"] = df.tmean_c_hour - df.tmean_c_hour.mean()
    return df


def build_processed_temperature() -> pd.DataFrame:
    ind = pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")
    temp = pd.read_parquet(GENERATED_DATA / "10_daily_temperature_kab.parquet")
    temp["date"] = pd.to_datetime(temp.date)
    temp = add_daily_features(temp)
    out = merge_daily(ind, temp)
    out = add_hourly_temperature(out)
    out = out[list(PROCESSED_TEMPERATURE_SCHEMA.columns)]
    return PROCESSED_TEMPERATURE_SCHEMA.validate(out)


def main() -> None:
    out = build_processed_temperature()
    output_path = GENERATED_DATA / "26_processed_temperature_data.parquet"
    out.to_parquet(output_path, index=False)
    log(f"wrote {len(out):,} rows to {output_path}")


if __name__ == "__main__":
    main()
