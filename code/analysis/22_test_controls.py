"""Robustness: do the Table 1 headline Heat x Stressor coefficients survive
the addition of rainfall and PM2.5 controls on the interview date?

Specifications run for each of the three stressors (Job loss / Palm shock / Fuel cut):
  (0) Baseline = headline spec from Table 1
  (1) + precip_mm
  (2) + pm25_ugm3
  (3) + precip_mm + pm25_ugm3

DV, controls, FE, SE clustering all match Table 1 exactly.

Console only.
"""
from __future__ import annotations

import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "data" / "generated"

sys.path.insert(0, str(PROJECT / "code" / "analysis"))
mod14 = import_module("14_unified_refined")
PALM_3MO_DECLINE = mod14.PALM_3MO_DECLINE


CONTROLS = "age + female + edu_yrs + married + widowed"
FE_POOLED = "month + year + wave + kab_code"
FE_IFLS5  = "month + year + kab_code"


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(OUT / "analysis_dataset.parquet")
    fin  = pd.read_parquet(OUT / "financial_shocks.parquet")
    fin2 = pd.read_parquet(OUT / "financial_shocks_v2.parquet")
    df = df.merge(fin[["pidlink", "wave", "job_loss_within_yr"]],
                  on=["pidlink", "wave"], how="left")
    df = df.merge(fin2[["pidlink", "wave", "palm_farmer_hh", "transport_share"]],
                  on=["pidlink", "wave"], how="left")

    # Merge PM2.5 daily polygon-mean on (kab_code, interview_date)
    pm = pd.read_parquet(OUT / "pm25_daily_kab.parquet")
    pm["date"] = pd.to_datetime(pm.date)
    df = df.merge(
        pm[["kab_code", "date", "pm25_ugm3"]],
        left_on=["kab_code", "interview_date"], right_on=["kab_code", "date"],
        how="left",
    ).drop(columns=["date"])

    df["female"] = (df.sex == "F").astype(int)
    df = df.dropna(subset=[
        "cesd_raw", "tmean_c", "kab_code", "month", "year", "wave",
        "age", "female", "edu_yrs", "married", "widowed",
        "job_loss_within_yr", "palm_farmer_hh", "transport_share",
        "interview_date", "precip_mm",
    ])
    df["cesd_z"] = df.groupby("wave")["cesd_raw"].transform(lambda s: (s - s.mean()) / s.std())
    df["heat_c_dev"] = df.tmean_c - df.tmean_c.mean()
    df["intvw_ym"] = list(zip(df.interview_date.dt.year, df.interview_date.dt.month))
    df["palm_3mo_decline"] = df.intvw_ym.map(PALM_3MO_DECLINE).fillna(0.0)
    df["palm_shock"] = df.palm_farmer_hh * df.palm_3mo_decline
    df["post_subsidy"] = (df.interview_date >= pd.Timestamp("2014-11-18")).astype(int)
    df["fuel_shock"] = df.post_subsidy * df.transport_share

    counts = df.kab_code.value_counts()
    df = df[df.kab_code.isin(counts[counts > 1].index)].copy()
    return df


def stars(p: float) -> str:
    if pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def fit(df: pd.DataFrame, stressor: str, extra_main_control: str, fe: str,
        added_controls: str) -> dict:
    """Run Heat * Stressor with possibly added weather/pollution controls."""
    extras = (" + " + added_controls) if added_controls else ""
    formula = (
        f"cesd_z ~ heat_c_dev * {stressor}{extra_main_control} + {CONTROLS}{extras} | {fe}"
    )
    m = pf.feols(formula, data=df, vcov={"CRV1": "kab_code"})
    inter = f"heat_c_dev:{stressor}"
    coefs = m.coef()
    return {
        "n":         int(m._N),
        "inter_b":   float(coefs.get(inter, np.nan)),
        "inter_se":  float(m.se().get(inter, np.nan)),
        "inter_p":   float(m.pvalue().get(inter, np.nan)),
        "heat_b":    float(coefs.get("heat_c_dev", np.nan)),
        "heat_p":    float(m.pvalue().get("heat_c_dev", np.nan)),
        "precip_b":  float(coefs.get("precip_mm", np.nan)) if "precip_mm" in coefs.index else np.nan,
        "precip_p":  float(m.pvalue().get("precip_mm", np.nan)) if "precip_mm" in coefs.index else np.nan,
        "pm25_b":    float(coefs.get("pm25_ugm3", np.nan)) if "pm25_ugm3" in coefs.index else np.nan,
        "pm25_p":    float(m.pvalue().get("pm25_ugm3", np.nan)) if "pm25_ugm3" in coefs.index else np.nan,
    }


def main() -> None:
    df = load_data()
    n_total = len(df)
    n_pm = df.pm25_ugm3.notna().sum()
    print(f"Sample after merging precip + PM2.5: n={n_total:,}  "
          f"(PM2.5 coverage: {n_pm:,} = {100*n_pm/n_total:.1f}%)")

    # Cap PM2.5 outliers? Print the distribution first.
    print("\nPrecip (mm) distribution:")
    print(df.precip_mm.describe().round(2))
    print("\nPM2.5 (ug/m^3) distribution:")
    print(df.pm25_ugm3.describe().round(2))

    # For PM2.5-included specs we drop rows where PM2.5 is missing.
    df_pm = df.dropna(subset=["pm25_ugm3"]).copy()

    stressors = [
        ("Job loss",   "job_loss_within_yr", "",                           FE_POOLED, df,    df_pm),
        ("Palm shock", "palm_shock",         " + palm_farmer_hh",          FE_POOLED, df,    df_pm),
        ("Fuel shock", "fuel_shock",         " + transport_share",         FE_IFLS5,
            df[df.wave == "IFLS5"].copy(), df_pm[df_pm.wave == "IFLS5"].copy()),
    ]

    specs = [
        ("(0) Baseline",            ""),
        ("(1) + precip_mm",         "precip_mm"),
        ("(2) + pm25_ugm3",         "pm25_ugm3"),
        ("(3) + precip + pm25_ugm3","precip_mm + pm25_ugm3"),
    ]

    for label, stressor_var, extra_main, fe, dfA, dfA_pm in stressors:
        print("\n" + "=" * 105)
        print(f"  {label}  (stressor = {stressor_var})")
        print("=" * 105)
        print(f"{'spec':<32}{'Heat x Stress':>18}{'(SE)':>10}{'p':>8}"
              f"{'precip beta':>16}{'pm25 beta':>14}{'n':>10}")
        for spec_label, extras in specs:
            d = dfA_pm if "pm25" in extras else dfA
            r = fit(d, stressor_var, extra_main, fe, extras)
            line = (f"{spec_label:<32}"
                    f"{r['inter_b']:>+12.4f}{stars(r['inter_p']):<3}"
                    f" {('('+f'{r['inter_se']:.4f}'+')'):>9}"
                    f" {r['inter_p']:>7.3f}")
            if "precip" in extras:
                line += f" {r['precip_b']:>+11.5f}{stars(r['precip_p']):<3}"
            else:
                line += f"{'':>14}"
            if "pm25" in extras:
                line += f" {r['pm25_b']:>+10.5f}{stars(r['pm25_p']):<3}"
            else:
                line += f"{'':>13}"
            line += f" {r['n']:>9,}"
            print(line)

    print("\nNotes:")
    print("  Heat x Stressor units = SD of CES-D per (degree C x stressor unit).")
    print("  precip_mm and pm25_ugm3 both measured on the interview date for that kabupaten.")
    print("  Specs (2) and (3) restrict the sample to interview dates where PM2.5 is observed.")


if __name__ == "__main__":
    main()
