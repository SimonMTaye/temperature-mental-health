"""Build person-wave processed temperature variables for analysis tables.

Output: data/generated/26_processed_temperature_data.parquet
Row level: one person-wave record, keyed by pidlink + wave.
"""

import numpy as np
import pandas as pd

from data._schemas import PROCESSED_TEMPERATURE_SCHEMA
from data.config import GENERATED_DATA
from library.log import log


POST_SUBSIDY_DATE = pd.Timestamp("2014-11-18")
PAST30_TEMP_BIN_WIDTH = 1.5


TEMP_QUINTILE_LABELS = [f"q{i}" for i in range(1, 6)]
TEMP_DECILE_LABELS = [f"d{i}" for i in range(1, 11)]


def _utc_offset_hours(province_code: int) -> int:
    if (
        (11 <= province_code <= 18)
        or (21 <= province_code <= 36)
        or province_code in (61, 62)
    ):
        return 7
    if (
        (51 <= province_code <= 53)
        or (63 <= province_code <= 65)
        or (71 <= province_code <= 76)
    ):
        return 8
    if (81 <= province_code <= 82) or (91 <= province_code <= 94):
        return 9
    return 7


def _rolling_mean_excluding_today(
    s: pd.Series, window: int, min_periods: int
) -> pd.Series:
    return s.rolling(window, min_periods=min_periods).mean().shift(1)


def add_past30_bin_counts(
    temp: pd.DataFrame,
    *,
    source_col: str,
) -> pd.DataFrame:
    """Count prior-30-day local temperatures in percentile-aligned 1.5 C bins."""
    p5, p95 = temp[source_col].dropna().quantile([0.05, 0.95])
    last_edge = np.floor(p95 / PAST30_TEMP_BIN_WIDTH) * PAST30_TEMP_BIN_WIDTH
    first_edge = last_edge - (
        round((last_edge - p5) / PAST30_TEMP_BIN_WIDTH) * PAST30_TEMP_BIN_WIDTH
    )
    n_edges = int(round((last_edge - first_edge) / PAST30_TEMP_BIN_WIDTH)) + 1
    edges = np.linspace(
        first_edge,
        last_edge,
        n_edges,
    )
    ranges = [(-np.inf, edges[0])]
    ranges.extend(zip(edges[:-1], edges[1:], strict=True))
    ranges.append((edges[-1], np.inf))

    bin_names = []
    for lower, upper in ranges:
        if np.isneginf(lower):
            upper_sign = "m" if upper < 0 else ""
            upper_suffix = f"{upper_sign}{abs(upper):.1f}".replace(".", "p")
            suffix = f"lt_{upper_suffix}"
        elif np.isposinf(upper):
            lower_sign = "m" if lower < 0 else ""
            lower_suffix = f"{lower_sign}{abs(lower):.1f}".replace(".", "p")
            suffix = f"gt_{lower_suffix}"
        else:
            lower_sign = "m" if lower < 0 else ""
            lower_suffix = f"{lower_sign}{abs(lower):.1f}".replace(".", "p")
            upper_sign = "m" if upper < 0 else ""
            upper_suffix = f"{upper_sign}{abs(upper):.1f}".replace(".", "p")
            suffix = f"{lower_suffix}_{upper_suffix}"
        name = f"{source_col}_past30_{suffix}"
        bin_names.append(name)
        in_bin = temp[source_col].ge(lower) & temp[source_col].lt(upper)
        if np.isneginf(lower):
            in_bin = temp[source_col].lt(upper)
        if np.isposinf(upper):
            in_bin = temp[source_col].ge(lower)
        temp[name] = (
            in_bin.astype(float)
            .where(temp[source_col].notna())
            .groupby(temp["gadm_fullcode"])
            .transform(lambda s: s.shift(1).rolling(30).sum())
        )
    log(f"built {len(bin_names)} past-30-day bins for {source_col}: {bin_names}")
    return temp


def add_temperature_quantile_day_counts(
    temp: pd.DataFrame,
    *,
    source_col: str = "tmean_c",
) -> pd.DataFrame:
    """Count past 30/7 daily mean-temperature days in raw-data quantile bins."""
    temp = temp.sort_values(["gadm_fullcode", "date"], kind="stable").copy()
    series = temp[source_col].dropna()
    quantile_specs = [
        ("q", TEMP_QUINTILE_LABELS, [0.2, 0.4, 0.6, 0.8]),
        ("d", TEMP_DECILE_LABELS, [i / 10 for i in range(1, 10)]),
    ]
    for prefix, labels, probs in quantile_specs:
        cutpoints = series.quantile(probs).to_list()
        bins = [-np.inf, *cutpoints, np.inf]
        temp_bin = pd.cut(
            temp[source_col], bins=bins, labels=labels, include_lowest=True
        )
        rounded = ", ".join(
            f"{p:.1f}={v:.2f}" for p, v in zip(probs, cutpoints, strict=True)
        )
        log(f"{source_col} raw-data {prefix} cutpoints: {rounded}")
        for label in labels:
            in_bin = temp_bin.eq(label).astype(float).where(temp[source_col].notna())
            for window in (30, 7):
                temp[f"temp_days_past{window}_{label}"] = (
                    in_bin.groupby(temp["gadm_fullcode"])
                    .transform(lambda s: s.shift(1).rolling(window).sum())
                )
    return temp


def saturation_vapor_pressure_pa(temp_c: pd.Series) -> pd.Series:
    """Saturation vapor pressure over water from NPL's Magnus equation."""
    # https://www.npl.co.uk/resources/q-a/dew-point-and-relative-humidity
    return np.exp(np.log(611.2) + (17.62 * temp_c) / (243.12 + temp_c))


def relative_humidity_from_dewpoint(
    temp_c: pd.Series, dewpoint_c: pd.Series
) -> pd.Series:
    """Relative humidity from air temperature and dewpoint via NPL Eq. 1/2."""
    actual_vapor_pressure = saturation_vapor_pressure_pa(dewpoint_c)
    saturation_vapor_pressure = saturation_vapor_pressure_pa(temp_c)
    return (100.0 * actual_vapor_pressure / saturation_vapor_pressure).clip(0, 100)


def wetbulb_stull_c(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """Stull (2011) wet-bulb approximation using temperature in C and RH percent."""
    # Stull 2011, Journal of Applied Meteorology and Climatology.
    # https://doi.org/10.1175/JAMC-D-11-0143.1
    return (
        temp_c * np.arctan(0.151977 * np.sqrt(rh_pct + 8.313659))
        + np.arctan(temp_c + rh_pct)
        - np.arctan(rh_pct - 1.676331)
        + 0.00391838 * rh_pct ** (3 / 2) * np.arctan(0.023101 * rh_pct)
        - 4.686035
    )


def load_hourly_temperature() -> pd.DataFrame:
    hourly_path = GENERATED_DATA / "11_hourly_temperature_kab.parquet"
    assert hourly_path.exists(), f"Missing hourly temperature file: {hourly_path}"

    hourly = pd.read_parquet(hourly_path)
    required = [
        "gadm_fullcode",
        "province_code",
        "datetime_utc",
        "tmean_c_hour",
        "dewp_c_hour",
    ]
    missing = sorted(set(required).difference(hourly.columns))
    assert not missing, f"11_hourly_temperature_kab.parquet missing columns: {missing}"
    hourly = hourly[required].copy()
    hourly["datetime_utc"] = pd.to_datetime(hourly.datetime_utc, utc=True)
    return hourly


def add_hourly_wetbulb(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add hourly wet-bulb temperature from hourly air temperature and dewpoint."""
    hourly = hourly.copy()
    rh_pct = relative_humidity_from_dewpoint(
        hourly["tmean_c_hour"], hourly["dewp_c_hour"]
    )
    hourly["wetbulb_c_hour"] = wetbulb_stull_c(hourly["tmean_c_hour"], rh_pct)
    return hourly


def build_daily_wetbulb(hourly: pd.DataFrame) -> pd.DataFrame:
    """Compute local-day wet-bulb means from hourly temperature and dewpoint."""
    hourly = hourly.copy()
    utc_offset = hourly["province_code"].map(_utc_offset_hours)
    hourly["date"] = (
        hourly["datetime_utc"]
        .add(pd.to_timedelta(utc_offset, unit="h"))
        .dt.tz_localize(None)
        .dt.normalize()
    )
    daily = (
        hourly.groupby(["gadm_fullcode", "date"], as_index=False)["wetbulb_c_hour"]
        .mean()
        .rename(columns={"wetbulb_c_hour": "wetbulb_c"})
    )
    log(f"built daily wet-bulb means for {len(daily):,} geography-days")
    return daily


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
    temp["tmean_30d"] = grouped["tmean_c"].transform(
        lambda s: s.rolling(30, min_periods=15).mean()
    )
    temp["tmin_7d"] = grouped["tmin_c"].transform(
        lambda s: s.rolling(7, min_periods=4).mean()
    )
    temp["wetbulb_7d"] = grouped["wetbulb_c"].transform(
        lambda s: s.rolling(7, min_periods=4).mean()
    )
    temp["wetbulb_30d"] = grouped["wetbulb_c"].transform(
        lambda s: s.rolling(30, min_periods=15).mean()
    )
    temp["hot30_7d"] = (
        temp["tmax_c"]
        .gt(30.0)
        .astype(np.int8)
        .groupby(temp["gadm_fullcode"])
        .transform(lambda s: s.rolling(7, min_periods=4).sum())
    )
    p90 = temp.groupby("gadm_fullcode")["tmean_c"].transform(lambda s: s.quantile(0.90))
    temp["heatwave_7d"] = (
        temp["tmean_c"]
        .gt(p90)
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
        "tmean_30d",
        "tmin_7d",
        "wetbulb_c",
        "wetbulb_7d",
        "wetbulb_30d",
        "hot30_7d",
        "heatwave_7d",
    ]
    keep.extend(
        sorted(
            col
            for col in temp
            if col.startswith("tmean_c_past30_")
            or col.startswith("wetbulb_c_past30_")
            or col.startswith("temp_days_past")
        )
    )
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
    for col in [
        "tmean_c",
        "tmax_c",
        "tmin_c",
        "tmean_7d",
        "tmean_30d",
        "tmin_7d",
        "wetbulb_c",
        "wetbulb_7d",
        "wetbulb_30d",
        "hot30_7d",
        "heatwave_7d",
    ]:
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


def add_hourly_temperature(df: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hourly = hourly[
        [
            col
            for col in [
                "gadm_fullcode",
                "datetime_utc",
                "tmean_c_hour",
                "wetbulb_c_hour",
            ]
            if col in hourly.columns
        ]
    ]
    # ERA5 hourly timestamps are UTC; IFLS interview hours are local Indonesian time.
    df["utc_offset"] = df.province_code.map(_utc_offset_hours)
    local_hour = df.hour_start.round().clip(lower=0, upper=23).astype(int)
    df["interview_dt_utc"] = df["interview_date"].dt.tz_localize(
        "UTC"
    ) + pd.to_timedelta(local_hour - df["utc_offset"], unit="h")
    df = df.merge(
        hourly,
        left_on=["gadm_fullcode", "interview_dt_utc"],
        right_on=["gadm_fullcode", "datetime_utc"],
        how="left",
        validate="m:1",
    )
    df = df.drop(columns=["datetime_utc", "interview_dt_utc", "utc_offset"])
    matched = df.tmean_c_hour.notna().sum()
    log(f"matched hourly temperature for {matched:,} of {len(df):,} person-wave rows")
    df["heat_hr_dev"] = df.tmean_c_hour - df.tmean_c_hour.mean()
    df["wetbulb_hr_dev"] = df.wetbulb_c_hour - df.wetbulb_c_hour.mean()
    return df


def build_processed_temperature() -> pd.DataFrame:
    ind = pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")
    temp = pd.read_parquet(GENERATED_DATA / "10_daily_temperature_kab.parquet")
    hourly = add_hourly_wetbulb(load_hourly_temperature())
    wetbulb_daily = build_daily_wetbulb(hourly)
    temp["date"] = pd.to_datetime(temp.date)
    temp = temp.merge(
        wetbulb_daily, on=["gadm_fullcode", "date"], how="left", validate="1:1"
    )
    temp = add_daily_features(temp)
    temp = add_temperature_quantile_day_counts(temp)
    temp = add_past30_bin_counts(temp, source_col="tmean_c")
    temp = add_past30_bin_counts(temp, source_col="wetbulb_c")
    out = merge_daily(ind, temp)
    out = add_hourly_temperature(out, hourly)
    out = out[list(PROCESSED_TEMPERATURE_SCHEMA.columns)]
    return PROCESSED_TEMPERATURE_SCHEMA.validate(out)


def main() -> None:
    out = build_processed_temperature()
    output_path = GENERATED_DATA / "26_processed_temperature_data.parquet"
    out.to_parquet(output_path, index=False)
    log(f"wrote {len(out):,} rows to {output_path}")


if __name__ == "__main__":
    main()
