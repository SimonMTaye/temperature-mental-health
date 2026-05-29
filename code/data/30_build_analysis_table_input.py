"""Build the canonical analysis input.

This is the single downstream-facing person-wave artifact consumed by analysis,
table, and figure scripts.

Output: data/generated/30_analysis_table_input.parquet
"""
import numpy as np
import pandas as pd

from config import GENERATED_DATA
from _commodity_prices import COFFEE_PRICE, PALM_3MO_DECLINE, RUBBER_PRICE
from _schemas import ANALYSIS_TABLE_INPUT_SCHEMA

POST_SUBSIDY_DATE = pd.Timestamp("2014-11-18")
HAZE_MONTHS = {(2015, 9), (2015, 10), (2015, 11)}
CESD_FACTOR_COLUMNS = [
    "somatic",
    "depraffect",
    "posaffect",
    "somatic_z",
    "depraffect_z",
    "posaffect_z",
]


def ensure_cesd_factor_columns(ces: pd.DataFrame) -> pd.DataFrame:
    """Keep old CES-D score artifacts readable while 24 owns factor construction."""
    missing = [col for col in CESD_FACTOR_COLUMNS if col not in ces.columns]
    if missing:
        print(
            "warning: 24_cesd_scores.parquet is missing CES-D factor columns; "
            "rerun code/data/24_score_cesd.py when raw IFLS files are mounted"
        )
        ces = ces.copy()
        for col in missing:
            ces[col] = float("nan")
    return ces


def add_temp_lags(ind: pd.DataFrame, temp: pd.DataFrame) -> pd.DataFrame:
    """For each individual, attach temperature on interview_date plus lags/leads."""
    temp = temp.sort_values(["kabupaten_code", "date"]).copy()

    temp["tmean_lag1"] = temp.groupby("kabupaten_code")["tmean_c"].shift(1)
    temp["tmean_lag3"] = temp.groupby("kabupaten_code")["tmean_c"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
    temp["tmean_lag7"] = temp.groupby("kabupaten_code")["tmean_c"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
    temp["tmin_lag1"] = temp.groupby("kabupaten_code")["tmin_c"].shift(1)
    temp["tmax_lag1"] = temp.groupby("kabupaten_code")["tmax_c"].shift(1)
    temp["heat_idx_lag1"] = temp.groupby("kabupaten_code")["heat_idx_c"].shift(1)
    temp["tmean_base30"] = temp.groupby("kabupaten_code")["tmean_c"].rolling(30, min_periods=15).mean().reset_index(level=0, drop=True).shift(1)
    temp["tmean_lead7"] = temp.groupby("kabupaten_code")["tmean_c"].shift(-7)

    keep = [
        "kabupaten_code", "date",
        "tmean_c", "tmax_c", "tmin_c", "heat_idx_c", "rh_pct", "precip_mm",
        "tmean_lag1", "tmean_lag3", "tmean_lag7",
        "tmin_lag1", "tmax_lag1", "heat_idx_lag1",
        "tmean_base30", "tmean_lead7",
    ]
    out = ind.merge(
        temp[keep],
        left_on=["kabupaten_code", "interview_date"],
        right_on=["kabupaten_code", "date"],
        how="left",
    ).drop(columns=["date"])

    out["t_anom_today"] = out.tmean_c - out.tmean_base30
    out["t_anom_lag1"] = out.tmean_lag1 - out.tmean_base30
    return out


def _pce_decline(df: pd.DataFrame) -> pd.DataFrame:
    """Bottom-quartile of inter-wave PCE change for panel respondents."""
    pce = df[["pidlink", "wave", "pce"]].dropna(subset=["pce"])
    pce = pce[pce.pce > 0]
    pce_w = pce.pivot_table(index="pidlink", columns="wave", values="pce", aggfunc="mean")
    if "IFLS4" not in pce_w.columns or "IFLS5" not in pce_w.columns:
        return pd.DataFrame(columns=["pidlink", "wave", "pce_decline_q4"])
    panel = pce_w.dropna()
    # Approximate IDR inflation factor from 2007 to 2014 ≈ 1.7 (CPI rose ~70 %).
    # Real change = PCE_5 - PCE_4 * 1.7
    DEFLATOR = 1.7
    panel["pce_chg_real"] = panel["IFLS5"] - panel["IFLS4"] * DEFLATOR
    panel["pce_pct_chg"] = panel["pce_chg_real"] / (panel["IFLS4"] * DEFLATOR)
    thr = panel["pce_pct_chg"].quantile(0.25)
    panel["pce_decline_q4_flag"] = (panel["pce_pct_chg"] <= thr).astype(int)

    # Broadcast the same flag to BOTH wave-rows for each panel respondent
    panel = panel.reset_index()[["pidlink", "pce_decline_q4_flag"]]
    out = pd.concat([
        panel.assign(wave="IFLS4"),
        panel.assign(wave="IFLS5"),
    ], ignore_index=True)
    out = out.rename(columns={"pce_decline_q4_flag": "pce_decline_q4"})
    return out


def build_core_panel() -> pd.DataFrame:
    """Merge person, CES-D, covariate, and interview-date weather inputs."""
    ind = pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")
    ces = ensure_cesd_factor_columns(pd.read_parquet(GENERATED_DATA / "24_cesd_scores.parquet"))
    stress = pd.read_parquet(GENERATED_DATA / "22_stressors.parquet")
    temp = pd.read_parquet(GENERATED_DATA / "10_daily_temperature_kab.parquet")
    temp["date"] = pd.to_datetime(temp.date)

    print(f"individuals={len(ind):,}  cesd={len(ces):,}  stressors={len(stress):,}  temp={len(temp):,}")
    df = ind.merge(ces, on=["pidlink", "wave"], how="inner")
    df = df.merge(stress.drop(columns=["hhid"], errors="ignore"), on=["pidlink", "wave"], how="left")
    print(f"after CES-D merge: {len(df):,}")

    df = add_temp_lags(df, temp)
    df = df[df.age >= 15].copy()
    df = df.dropna(subset=["cesd_raw", "tmean_c"]).copy()
    print(f"core analysis sample: {len(df):,} adults")

    df["post_subsidy"] = (df.interview_date >= POST_SUBSIDY_DATE).astype(int)
    df["haze_2015"] = df.interview_date.apply(lambda d: int((d.year, d.month) in HAZE_MONTHS))
    df["yogya_quake_catchment"] = (df.wave.eq("IFLS4") & df.province_code.isin([33, 34])).astype(int)
    df["heat_bin"] = pd.cut(
        df.tmean_c,
        bins=[-np.inf, 22, 24, 26, 28, np.inf],
        labels=["<22", "22-24", "24-26", "26-28", "28+"],
    )
    df["month_year"] = df.interview_date.dt.to_period("M").astype(str)
    df["month"] = df.interview_date.dt.month
    df["year"] = df.interview_date.dt.year
    return df


def merge_optional_sidecar(
    df: pd.DataFrame,
    path_name: str,
    fill_zero_cols: list[str],
    *,
    drop_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Left-merge an optional sidecar and fill indicator/count columns with zero."""
    path = GENERATED_DATA / path_name
    if path.exists():
        sidecar = pd.read_parquet(path)
        if drop_cols:
            sidecar = sidecar.drop(columns=drop_cols, errors="ignore")
        df = df.merge(sidecar, on=["pidlink", "wave"], how="left")
    for col in fill_zero_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)
    return df


def add_model_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add common outcome, heat, threshold, and shock variables used by tables."""
    df = df.copy()
    df["precip_mm"] = df.precip_mm.clip(lower=0)
    df["female"] = (df.sex == "F").astype(int)
    df["cesd_z"] = df.groupby("wave")["cesd_raw"].transform(lambda s: (s - s.mean()) / s.std())

    for col in ["tmean_c", "tmax_c", "tmin_c"]:
        df[f"{col}_dev"] = df[col] - df[col].mean()
    df["heat_c_dev"] = df["tmean_c_dev"]

    df["cdd_tmax30"] = (df.tmax_c - 30.0).clip(lower=0)
    df["cdd_tmax32"] = (df.tmax_c - 32.0).clip(lower=0)
    df["cdd_tmin23"] = (df.tmin_c - 23.0).clip(lower=0)
    df["cdd_tmin24"] = (df.tmin_c - 24.0).clip(lower=0)

    intvw_ym = list(zip(df.interview_date.dt.year, df.interview_date.dt.month))
    df["palm_3mo_decline"] = pd.Series(intvw_ym, index=df.index).map(PALM_3MO_DECLINE).fillna(0.0)
    df["palm_shock"] = df.palm_farmer_hh * df.palm_3mo_decline
    df["fuel_shock"] = df.post_subsidy * df.transport_share

    rp = pd.Series(list(RUBBER_PRICE.values()))
    df["rubber_price_usd_kg"] = pd.Series(intvw_ym, index=df.index).map(RUBBER_PRICE)
    df["rubber_price_z"] = (df.rubber_price_usd_kg - rp.mean()) / rp.std()
    df["rubber_shock"] = df.rubber_farmer_individual * (-df.rubber_price_z.fillna(0)).clip(lower=0)

    cp = pd.Series(list(COFFEE_PRICE.values()))
    df["coffee_price_clb"] = pd.Series(intvw_ym, index=df.index).map(COFFEE_PRICE)
    df["coffee_price_z"] = (df.coffee_price_clb - cp.mean()) / cp.std()
    df["coffee_shock"] = df.coffee_farmer_individual * (-df.coffee_price_z.fillna(0)).clip(lower=0)
    return df


def main() -> None:
    GENERATED_DATA.mkdir(parents=True, exist_ok=True)

    df = build_core_panel()
    economic = pd.read_parquet(GENERATED_DATA / "20_economic_exposures.parquet")
    commodity_transport = pd.read_parquet(
        GENERATED_DATA / "25_commodity_transport_exposures.parquet"
    )

    df = df.merge(economic, on=["pidlink", "wave"], how="left")
    df = df.merge(
        commodity_transport,
        on=["pidlink", "wave"],
        how="left",
        suffixes=("", "_commodity_transport"),
    )
    if "palm_region_commodity_transport" in df.columns:
        df = df.drop(columns=["palm_region_commodity_transport"])

    df = merge_optional_sidecar(
        df,
        "21_health_bereavement_shocks.parquet",
        [
            "n_symptoms",
            "many_symptoms",
            "recent_hospitalised",
            "recent_accident_2y",
            "recently_widowed_5y",
        ],
    )
    df = merge_optional_sidecar(
        df,
        "23_finance_distress_shocks.parquet",
        ["debt_q4", "high_med_oop"],
        drop_cols=["hhid", "debt", "med_oop"],
    )

    pce_d = _pce_decline(df)
    df = df.merge(pce_d, on=["pidlink", "wave"], how="left")
    df["pce_decline_q4"] = df["pce_decline_q4"].fillna(0).astype(int)

    df = add_model_variables(df)
    df = df.dropna(
        subset=[
            "job_loss_within_yr",
            "palm_farmer_individual",
            "palm_farmer_hh",
            "rubber_farmer_individual",
            "coffee_farmer_individual",
            "transport_share",
            "vehicle_owner",
        ]
    ).copy()

    int_cols = [
        "job_loss_within_yr",
        "palm_farmer_individual",
        "palm_farmer_hh",
        "rubber_farmer_individual",
        "coffee_farmer_individual",
        "vehicle_owner",
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)

    df = df[list(ANALYSIS_TABLE_INPUT_SCHEMA.columns)]
    df = ANALYSIS_TABLE_INPUT_SCHEMA.validate(df)
    out_path = GENERATED_DATA / "30_analysis_table_input.parquet"
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df):,} rows to {out_path}")
    print(
        df.groupby("wave").agg(
            n=("pidlink", "size"),
            cesd_z_mean=("cesd_z", "mean"),
            job_loss_pct=("job_loss_within_yr", lambda s: 100 * s.mean()),
            palm_shock_mean=("palm_shock", "mean"),
            fuel_shock_mean=("fuel_shock", "mean"),
        ).round(4)
    )


if __name__ == "__main__":
    main()
