"""Replicate the fuel-cut column of Table 1 using motor-vehicle ownership
as the exposure measure instead of transport-spending share.

Rationale: transport_share is potentially endogenous (depressed people may
travel less; post-cut adjustment contaminates the post-period measure).
Vehicle ownership is pre-determined at the household level and directly
proxies fuel exposure: households without a motor vehicle consume
essentially no fuel and are unaffected by the cut.

Builds:
  owns_vehicle_hh_i = 1 if household reports owning at least one
                      vehicle (asset type 'E' in IFLS4 + IFLS5 b2_hr1
                      — 'Vehicles' includes cars and motorbikes).

Compares three specifications for the fuel-cut column (IFLS5 only, same
FE / controls / SE as Table 1):
  (A) Headline             : Heat x (post x transport_share)
  (B) Vehicle ownership    : Heat x (post x owns_vehicle)
  (C) Vehicle + transport  : both interactions in the same regression

Console output only.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "data" / "generated"
RAW = PROJECT.parent / "hao" / "IFLS"

from _table_input import load_table_input

CONTROLS = "age + female + edu_yrs + married + widowed"
FE_IFLS5 = "month + year + kabupaten_code"
CUT_DATE = pd.Timestamp("2014-11-18")


def load_vehicle_ownership() -> pd.DataFrame:
    """Pool IFLS4 + IFLS5 household-level vehicle ownership from b2_hr1.

    Asset type 'E' = 'Vehicles' (cars / motorbikes / bicycles); hr01 == 1
    means the household owns at least one such asset.
    """
    frames = []
    for tag, path in [("IFLS4", RAW / "IFLS4" / "hh07" / "b2_hr1.dta"),
                      ("IFLS5", RAW / "IFLS5" / "hh14" / "b2_hr1.dta")]:
        d = pd.read_stata(path, convert_categoricals=False)
        # ownership rows for asset type E
        d = d[d.hrtype == "E"].copy()
        # 1 = Yes, 3 = No, 9 = Missing in IFLS coding
        d["owns_vehicle"] = (d.hr01 == 1).astype(int)
        hhcol = "hhid07" if tag == "IFLS4" else "hhid14"
        d = d[[hhcol, "owns_vehicle"]].rename(columns={hhcol: "hhid"})
        d["wave"] = tag
        frames.append(d)
    veh = pd.concat(frames, ignore_index=True)
    return veh


def load_data() -> pd.DataFrame:
    df = load_table_input()
    df["owns_vehicle"] = df["vehicle_owner"]
    print("Vehicle ownership loaded from analysis_table_input.parquet")

    df = df.dropna(subset=[
        "cesd_raw", "tmean_c", "kabupaten_code", "month", "year", "wave",
        "age", "female", "edu_yrs", "married", "widowed",
        "job_loss_within_yr", "palm_farmer_hh", "transport_share",
        "interview_date", "owns_vehicle",
    ])
    df["fuel_shock_vehicle"] = df.post_subsidy * df.owns_vehicle

    counts = df.kabupaten_code.value_counts()
    df = df[df.kabupaten_code.isin(counts[counts > 1].index)].copy()
    return df


def stars(p):
    if pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def fit_one_inter(df, stressor, extra_main, exposed_value):
    """Return interaction coef + delta-method heat slope at exposed_value."""
    formula = f"cesd_z ~ heat_c_dev * {stressor}{extra_main} + {CONTROLS} | {FE_IFLS5}"
    m = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_code"})
    inter = f"heat_c_dev:{stressor}"
    coefs = m.coef()
    V = pd.DataFrame(m._vcov, index=coefs.index, columns=coefs.index)
    a = pd.Series(0.0, index=coefs.index)
    a["heat_c_dev"] = 1.0
    a[inter] = exposed_value
    slope = float(a @ coefs)
    slope_se = float(np.sqrt(a @ V @ a))
    slope_t = slope / slope_se if slope_se > 0 else np.nan
    slope_p = 2 * (1 - scipy_stats.norm.cdf(abs(slope_t)))
    return {
        "n":         int(m._N),
        "inter_b":   float(coefs.get(inter, np.nan)),
        "inter_se":  float(m.se().get(inter, np.nan)),
        "inter_p":   float(m.pvalue().get(inter, np.nan)),
        "stress_b":  float(coefs.get(stressor, np.nan)),
        "stress_se": float(m.se().get(stressor, np.nan)),
        "stress_p":  float(m.pvalue().get(stressor, np.nan)),
        "heat_b":    float(coefs.get("heat_c_dev", np.nan)),
        "heat_se":   float(m.se().get("heat_c_dev", np.nan)),
        "heat_p":    float(m.pvalue().get("heat_c_dev", np.nan)),
        "slope_b":   slope,
        "slope_se":  slope_se,
        "slope_p":   slope_p,
        "exposed_value": exposed_value,
    }


def fit_both(df):
    """Spec C: both interactions in the same regression."""
    formula = (f"cesd_z ~ heat_c_dev * fuel_shock + heat_c_dev * fuel_shock_vehicle "
               f"+ post_subsidy + transport_share + owns_vehicle "
               f"+ {CONTROLS} | {FE_IFLS5}")
    m = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_code"})
    coefs = m.coef()
    return {
        "n":  int(m._N),
        "fuel_inter_b":   float(coefs.get("heat_c_dev:fuel_shock", np.nan)),
        "fuel_inter_se":  float(m.se().get("heat_c_dev:fuel_shock", np.nan)),
        "fuel_inter_p":   float(m.pvalue().get("heat_c_dev:fuel_shock", np.nan)),
        "veh_inter_b":    float(coefs.get("heat_c_dev:fuel_shock_vehicle", np.nan)),
        "veh_inter_se":   float(m.se().get("heat_c_dev:fuel_shock_vehicle", np.nan)),
        "veh_inter_p":    float(m.pvalue().get("heat_c_dev:fuel_shock_vehicle", np.nan)),
    }


def main():
    df = load_data()
    sub5 = df[df.wave == "IFLS5"].copy()
    counts = sub5.kabupaten_code.value_counts()
    sub5 = sub5[sub5.kabupaten_code.isin(counts[counts > 1].index)].copy()
    print(f"IFLS5 sample (after singleton-kab restriction): n={len(sub5):,}")
    print(f"  owns_vehicle share: {sub5.owns_vehicle.mean()*100:.1f}%")
    print(f"  post_subsidy share: {sub5.post_subsidy.mean()*100:.1f}%")
    print(f"  post & owns_vehicle: {(sub5.post_subsidy * sub5.owns_vehicle).mean()*100:.1f}%")
    print(f"  transport_share mean: {sub5.transport_share.mean():.3f}, p75: {sub5.transport_share.quantile(0.75):.3f}")

    print("\n" + "=" * 100)
    print("Fuel-cut column — three exposure measures")
    print("=" * 100)

    print("\n(A) HEADLINE: fuel_shock = post_subsidy x transport_share")
    rA = fit_one_inter(sub5, "fuel_shock",
                       " + post_subsidy + transport_share",
                       exposed_value=0.10)
    print(f"  Heat x Fuel:        beta = {rA['inter_b']:+.4f}{stars(rA['inter_p'])}  (SE {rA['inter_se']:.4f}, p={rA['inter_p']:.3f})")
    print(f"  Heat (at S=0):      beta = {rA['heat_b']:+.4f}{stars(rA['heat_p'])}  (SE {rA['heat_se']:.4f})")
    print(f"  Stressor:           beta = {rA['stress_b']:+.4f}{stars(rA['stress_p'])}  (SE {rA['stress_se']:.4f})")
    print(f"  Heat slope|exposed: beta = {rA['slope_b']:+.4f}{stars(rA['slope_p'])}  [at fuel_shock=0.10, n={rA['n']:,}]")

    print("\n(B) VEHICLE: fuel_shock_vehicle = post_subsidy x owns_vehicle  (binary)")
    rB = fit_one_inter(sub5, "fuel_shock_vehicle",
                       " + post_subsidy + owns_vehicle",
                       exposed_value=1.0)
    print(f"  Heat x FuelVeh:     beta = {rB['inter_b']:+.4f}{stars(rB['inter_p'])}  (SE {rB['inter_se']:.4f}, p={rB['inter_p']:.3f})")
    print(f"  Heat (at S=0):      beta = {rB['heat_b']:+.4f}{stars(rB['heat_p'])}  (SE {rB['heat_se']:.4f})")
    print(f"  Stressor:           beta = {rB['stress_b']:+.4f}{stars(rB['stress_p'])}  (SE {rB['stress_se']:.4f})")
    print(f"  Heat slope|exposed: beta = {rB['slope_b']:+.4f}{stars(rB['slope_p'])}  [at owns_vehicle=1, n={rB['n']:,}]")

    print("\n(C) BOTH: include both interactions in same regression")
    rC = fit_both(sub5)
    print(f"  Heat x FuelShock (transport_share): beta = {rC['fuel_inter_b']:+.4f}{stars(rC['fuel_inter_p'])}  (SE {rC['fuel_inter_se']:.4f}, p={rC['fuel_inter_p']:.3f})")
    print(f"  Heat x FuelShock (vehicle):         beta = {rC['veh_inter_b']:+.4f}{stars(rC['veh_inter_p'])}  (SE {rC['veh_inter_se']:.4f}, p={rC['veh_inter_p']:.3f})")
    print(f"  n = {rC['n']:,}")


if __name__ == "__main__":
    main()
