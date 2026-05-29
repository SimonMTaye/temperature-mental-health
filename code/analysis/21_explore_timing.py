"""Exploration: timing variants for palm and fuel shocks.

For palm: cumulative % price decline over the past X months x palm-farmer HH,
varying X in {1, 3, 6}. (12-month requires extending the palm price series
back into 2006 / 2013, which we skip for now.)

For fuel: an indicator for "interviewed within X months of the 18 Nov 2014
fuel-subsidy cut" x transport-spending share, varying X in {1, 3, 6, 12}.
Sample is all IFLS5; the implied "control" includes pre-cut respondents
plus post-cut respondents interviewed beyond X months.

Reports the Heat x Stressor interaction and the heat slope at an exposed
reference value, for each window. Console only.
"""
from __future__ import annotations

import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "data" / "generated"

sys.path.insert(0, str(PROJECT / "code" / "analysis"))
mod14 = import_module("14_unified_refined")
PALM_PRICE_FULL = mod14.PALM_PRICE_FULL


CONTROLS = "age + female + edu_yrs + married + widowed"
FE_POOLED = "month + year + wave + kabupaten_code"
FE_IFLS5  = "month + year + kabupaten_code"
CUT_DATE  = pd.Timestamp("2014-11-18")


def palm_X_decline(X: int) -> dict:
    """Build (year, month) -> cumulative %-decline-over-prior-X-months (>=0)."""
    rows = sorted(PALM_PRICE_FULL.items())
    keys = [k for k, _ in rows]
    vals = [v for _, v in rows]
    decline = {}
    for i, (yr, mo) in enumerate(keys):
        if i < X:
            continue
        prev_yr, prev_mo = keys[i - X]
        gap = (yr - prev_yr) * 12 + (mo - prev_mo)
        if gap != X:
            continue
        chg = (vals[i] - vals[i - X]) / vals[i - X]
        decline[(yr, mo)] = max(-chg, 0.0)
    return decline


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(OUT / "analysis_dataset.parquet")
    fin  = pd.read_parquet(OUT / "financial_shocks.parquet")
    fin2 = pd.read_parquet(OUT / "financial_shocks_v2.parquet")
    df = df.merge(fin[["pidlink", "wave", "job_loss_within_yr"]],
                  on=["pidlink", "wave"], how="left")
    df = df.merge(fin2[["pidlink", "wave", "palm_farmer_hh", "transport_share"]],
                  on=["pidlink", "wave"], how="left")
    df["female"] = (df.sex == "F").astype(int)
    df = df.dropna(subset=[
        "cesd_raw", "tmean_c", "kabupaten_code", "month", "year", "wave",
        "age", "female", "edu_yrs", "married", "widowed",
        "job_loss_within_yr", "palm_farmer_hh", "transport_share",
        "interview_date",
    ])
    df["cesd_z"] = df.groupby("wave")["cesd_raw"].transform(lambda s: (s - s.mean()) / s.std())
    df["heat_c_dev"] = df.tmean_c - df.tmean_c.mean()
    df["intvw_ym"] = list(zip(df.interview_date.dt.year, df.interview_date.dt.month))
    counts = df.kabupaten_code.value_counts()
    df = df[df.kabupaten_code.isin(counts[counts > 1].index)].copy()
    return df


def stars(p: float) -> str:
    if pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def fit_interaction(df: pd.DataFrame, stressor_var: str, extra_control: str,
                    fe: str, exposed_value: float) -> dict:
    """Heat x Stressor interaction. Also returns heat slope at exposed_value."""
    formula = f"cesd_z ~ heat_c_dev * {stressor_var}{extra_control} + {CONTROLS} | {fe}"
    m = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_code"})
    inter = f"heat_c_dev:{stressor_var}"
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
        "n":        int(m._N),
        "inter_b":  float(coefs.get(inter, np.nan)),
        "inter_se": float(m.se().get(inter, np.nan)),
        "inter_p":  float(m.pvalue().get(inter, np.nan)),
        "stress_b":  float(coefs.get(stressor_var, np.nan)),
        "stress_se": float(m.se().get(stressor_var, np.nan)),
        "stress_p":  float(m.pvalue().get(stressor_var, np.nan)),
        "slope_b":  slope,
        "slope_se": slope_se,
        "slope_p":  slope_p,
        "exposed_value": exposed_value,
    }


def main() -> None:
    df = load_data()
    print(f"Pooled sample: n={len(df):,}  IFLS4: {(df.wave=='IFLS4').sum():,}  IFLS5: {(df.wave=='IFLS5').sum():,}")

    # -------- Palm: vary the cumulation window --------
    print("\n" + "=" * 100)
    print("PALM: cumulative % decline over preceding X months, x palm-farmer HH")
    print("=" * 100)
    palm_windows = [1, 3, 6]
    print(f"{'X (mo)':>6}  {'cov %':>6}  {'p75>0':>8}  {'exp@':>6}  "
          f"{'Heat x Palm':>16}  {'p':>8}  {'slope|exp':>14}  {'p':>8}  {'n':>8}")
    palm_rows = []
    for X in palm_windows:
        decline_map = palm_X_decline(X)
        d = df.copy()
        d["palm_X_decline"] = d.intvw_ym.map(decline_map)
        # Drop rows where the X-month decline can't be computed (early IFLS5 months)
        d = d.dropna(subset=["palm_X_decline"])
        cov = len(d) / len(df) * 100
        d["palm_shock_X"] = d.palm_farmer_hh * d.palm_X_decline
        # Exposed reference: 75th percentile of positive palm shocks (similar to figure 1)
        pos = d.loc[d.palm_shock_X > 0, "palm_shock_X"]
        p75_pos = pos.quantile(0.75) if len(pos) > 0 else np.nan
        r = fit_interaction(d, "palm_shock_X",
                            extra_control=" + palm_farmer_hh",
                            fe=FE_POOLED,
                            exposed_value=p75_pos)
        palm_rows.append({"X": X, "cov_pct": cov, "p75_pos": p75_pos, **r})
        print(f"{X:>6}  {cov:>5.1f}%  {p75_pos:>8.3f}  {p75_pos:>6.3f}  "
              f"{r['inter_b']:>+10.4f}{stars(r['inter_p']):>3}  ({r['inter_se']:.4f})"
              f"  {r['slope_b']:>+10.4f}{stars(r['slope_p']):>3}  ({r['slope_se']:.4f})"
              f"  {r['n']:>8,}")

    # -------- Fuel: vary how recent the cut is --------
    print("\n" + "=" * 100)
    print("FUEL: indicator for interviewed within X months of 18 Nov 2014 cut, x transport-share")
    print("=" * 100)
    fuel_windows = [1, 3, 6, 12]
    sub5 = df[df.wave == "IFLS5"].copy()
    counts5 = sub5.kabupaten_code.value_counts()
    sub5 = sub5[sub5.kabupaten_code.isin(counts5[counts5 > 1].index)].copy()
    print(f"IFLS5 sample (after singleton-kab restriction): n={len(sub5):,}")
    print(f"{'X (mo)':>6}  {'expos%':>7}  {'Heat x Fuel':>16}  {'p':>8}  "
          f"{'slope|p75':>14}  {'p':>8}  {'n':>8}")
    fuel_rows = []
    for X in fuel_windows:
        d = sub5.copy()
        cutoff = CUT_DATE + pd.DateOffset(months=X)
        d["post_within_X"] = ((d.interview_date >= CUT_DATE) & (d.interview_date < cutoff)).astype(int)
        d["fuel_shock_X"] = d.post_within_X * d.transport_share
        expos_pct = d.post_within_X.mean() * 100
        # Exposed reference: p75 of fuel_shock_X among exposed
        pos = d.loc[d.fuel_shock_X > 0, "fuel_shock_X"]
        p75_pos = pos.quantile(0.75) if len(pos) > 0 else np.nan
        r = fit_interaction(d, "fuel_shock_X",
                            extra_control=" + post_within_X + transport_share",
                            fe=FE_IFLS5,
                            exposed_value=p75_pos)
        fuel_rows.append({"X": X, "expos_pct": expos_pct, "p75_pos": p75_pos, **r})
        print(f"{X:>6}  {expos_pct:>6.1f}%  "
              f"{r['inter_b']:>+10.4f}{stars(r['inter_p']):>3}  ({r['inter_se']:.4f})"
              f"  {r['slope_b']:>+10.4f}{stars(r['slope_p']):>3}  ({r['slope_se']:.4f})"
              f"  {r['n']:>8,}")

    print("\nNotes:")
    print("  Palm:  cov% = share of pooled sample with X-month decline computable;")
    print("         exp@ = exposed reference value (p75 of positive palm_shock_X).")
    print("  Fuel:  expos% = share of IFLS5 sample within X months of the cut;")
    print("         slope|p75 = heat slope at p75 of fuel_shock_X among exposed.")
    print("  Interaction units differ across X for palm (longer window = larger declines),")
    print("  so the heat slope at exposed reference is the apples-to-apples comparison.")


if __name__ == "__main__":
    main()
