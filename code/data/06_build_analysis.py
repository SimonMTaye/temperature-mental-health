"""Merge everything into the analysis-ready dataset.

Combines:
  individuals.parquet           (pidlink, wave, interview_date, prov/kab/kec)
  cesd_scores.parquet           (pidlink, wave, cesd_raw, depressed)
  stressors.parquet             (pidlink, wave, age, sex, edu_yrs, married,
                                 widowed, hh_size, pce_log, pce_quintile,
                                 disaster_*, loan_rejected, agri_occupation)
  daily_temperature_kab.parquet (kab_code, date, tmean_c, tmax_c, tmin_c, ...)

Adds:
  Day-of-interview temperature: t_today_*, t_today_heat_idx
  Lagged temperature: t_lag1_*, t_lag3_*, t_lag7_*  (1, 3, 7 day lag means)
  Lead temperature: t_lead7_*                      (placebo)
  30-day baseline temperature: t_base30_*
  Macro shock flags:
      post_subsidy   = interview_date >= 2014-11-18  (Jokowi fuel-subsidy cut)
      haze_2015      = interview_date in 2015-09 / 2015-10 / 2015-11
      yogya_quake    = wave==IFLS4 & prov in {DIY, Jateng}     (post-quake catchment)

Output: data/generated/analysis_dataset.parquet  (one row per pidlink × wave; adults with CES-D)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "data" / "generated"

POST_SUBSIDY_DATE = pd.Timestamp("2014-11-18")
HAZE_MONTHS = {(2015, 9), (2015, 10), (2015, 11)}


def add_temp_lags(ind: pd.DataFrame, temp: pd.DataFrame) -> pd.DataFrame:
    """For each individual, attach temperature on interview_date + lags / leads."""
    temp = temp.sort_values(["kab_code", "date"]).copy()

    # Compute rolling means per kab_code (need indexed by date for asof-join lags)
    temp["tmean_lag1"] = temp.groupby("kab_code")["tmean_c"].shift(1)
    temp["tmean_lag3"] = temp.groupby("kab_code")["tmean_c"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
    temp["tmean_lag7"] = temp.groupby("kab_code")["tmean_c"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
    temp["tmin_lag1"] = temp.groupby("kab_code")["tmin_c"].shift(1)
    temp["tmax_lag1"] = temp.groupby("kab_code")["tmax_c"].shift(1)
    temp["heat_idx_lag1"] = temp.groupby("kab_code")["heat_idx_c"].shift(1)
    temp["tmean_base30"] = temp.groupby("kab_code")["tmean_c"].rolling(30, min_periods=15).mean().reset_index(level=0, drop=True).shift(1)
    temp["tmean_lead7"] = temp.groupby("kab_code")["tmean_c"].shift(-7)

    keep = [
        "kab_code", "date",
        "tmean_c", "tmax_c", "tmin_c", "heat_idx_c", "rh_pct", "precip_mm",
        "tmean_lag1", "tmean_lag3", "tmean_lag7",
        "tmin_lag1", "tmax_lag1", "heat_idx_lag1",
        "tmean_base30", "tmean_lead7",
    ]
    out = ind.merge(
        temp[keep],
        left_on=["kab_code", "interview_date"], right_on=["kab_code", "date"], how="left",
    ).drop(columns=["date"])

    # Heat-anomaly: today's temp minus 30-day baseline (within-kab deviation)
    out["t_anom_today"] = out.tmean_c - out.tmean_base30
    out["t_anom_lag1"] = out.tmean_lag1 - out.tmean_base30

    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ind = pd.read_parquet(OUT / "individuals.parquet")
    ces = pd.read_parquet(OUT / "cesd_scores.parquet")
    stress = pd.read_parquet(OUT / "stressors.parquet")
    temp = pd.read_parquet(OUT / "daily_temperature_kab.parquet")
    temp["date"] = pd.to_datetime(temp.date)
    print(f"individuals={len(ind):,}  cesd={len(ces):,}  stressors={len(stress):,}  temp={len(temp):,}")

    df = ind.merge(ces[["pidlink", "wave", "cesd_raw", "cesd10_count", "depressed", "n_items"]],
                   on=["pidlink", "wave"], how="inner")
    df = df.merge(stress.drop(columns=["hhid"], errors="ignore"),
                  on=["pidlink", "wave"], how="left")
    print(f"after CES-D merge: {len(df):,}")

    df = add_temp_lags(df, temp)

    # Sample restrictions
    df = df[df.age >= 15].copy()
    df = df.dropna(subset=["cesd_raw", "tmean_c"]).copy()
    print(f"final analysis sample: {len(df):,} adults")

    # Macro flags
    df["post_subsidy"] = (df.interview_date >= POST_SUBSIDY_DATE).astype(int)
    df["haze_2015"] = df.interview_date.apply(
        lambda d: int((d.year, d.month) in HAZE_MONTHS)
    )
    df["yogya_quake_catchment"] = (df.wave.eq("IFLS4") & df.prov_code.isin([33, 34])).astype(int)

    # Heat-stress derived: a few discrete heat bins for non-linear specs
    df["heat_bin"] = pd.cut(
        df.tmean_c,
        bins=[-np.inf, 22, 24, 26, 28, np.inf],
        labels=["<22", "22-24", "24-26", "26-28", "28+"],
    )
    df["month_year"] = df.interview_date.dt.to_period("M").astype(str)
    df["month"] = df.interview_date.dt.month
    df["year"] = df.interview_date.dt.year

    df.to_parquet(OUT / "analysis_dataset.parquet", index=False)
    print(f"\nwrote {len(df):,} rows to {OUT/'analysis_dataset.parquet'}")
    print("by wave:")
    print(df.groupby("wave").agg(
        n=("pidlink", "size"),
        cesd_mean=("cesd_raw", "mean"),
        depressed_pct=("depressed", lambda s: 100*s.mean()),
        tmean_mean=("tmean_c", "mean"),
        tmax_mean=("tmax_c", "mean"),
    ).round(2))


if __name__ == "__main__":
    main()
