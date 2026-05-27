"""Unified refined analysis addressing the v1-note caveats.

Three changes from the v1 analysis:

  (1) DV = z-score CES-D within wave  →  pools IFLS4 + IFLS5 cleanly despite
      different CES-D scoring (IFLS4 screener vs IFLS5 4-pt frequency).
      Coefficients now read in SDs of CES-D, comparable across waves.

  (2) Heat is mean-centered (tmean_c - sample_mean)  →  the main effect of the
      stressor in interaction models is now interpretable at AVERAGE temperature
      (not at the meaningless extrapolated tmean = 0°C).

  (3) Three refinements:
      - Job loss within 12 months: only this version (drop the 5-year version)
      - Palm: agricultural workers in palm regions × monthly palm price
              (individual-level, not just province-level)
      - Fuel subsidy: continuous transport-spending share (not binary urban)

  (4) lincom-style marginal effects reported for each interaction:
      - Effect of heat at stress=0 (and stress=1)
      - Effect of stress at heat = mean (and at mean ± 1 SD)
      with cluster-robust SE via delta method on the kab-clustered vcov.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "data" / "generated"
RES = OUT / "results"
RES.mkdir(parents=True, exist_ok=True)


# Extended palm price series (USD/MT, World Bank Pink Sheet) covering enough months
# either side of each fielding window so we can compute 3-month lagged price changes.
PALM_PRICE_FULL = {
    (2007, 1):780, (2007, 2):780, (2007, 3):770, (2007, 4):770, (2007, 5):810,
    (2007, 6):850, (2007, 7):879, (2007, 8):829, (2007, 9):866, (2007,10):861,
    (2007,11):950, (2007,12):1030,
    (2008, 1):1075,(2008, 2):1188,(2008, 3):1306,(2008, 4):1180,(2008, 5):1234,
    (2008, 6):1199,(2008, 7):1119,(2008, 8): 856,(2008, 9): 706,
    (2014, 5): 870,(2014, 6): 860,(2014, 7): 810,(2014, 8): 745,(2014, 9): 695,
    (2014,10): 696,(2014,11): 712,(2014,12): 715,
    (2015, 1): 678,(2015, 2): 651,(2015, 3): 657,(2015, 4): 660,(2015, 5): 656,
    (2015, 6): 658,(2015, 7): 627,(2015, 8): 528,(2015, 9): 511,(2015,10): 528,
    (2015,11): 529,(2015,12): 549,
}


def _palm_price_lookup() -> dict:
    """Build (year, month) -> %-decline-over-prior-3-months lookup. Drops only (≥0)."""
    rows = sorted(PALM_PRICE_FULL.items())  # (yr, mo) -> price
    keys = [k for k,_ in rows]
    vals = [v for _,v in rows]
    decline = {}
    for i, (yr, mo) in enumerate(keys):
        if i < 3:
            continue
        # Check that the 3-prior month is sequential (not across the 2009-2014 gap)
        prev_yr, prev_mo = keys[i-3]
        gap_months = (yr - prev_yr) * 12 + (mo - prev_mo)
        if gap_months != 3:
            continue
        chg = (vals[i] - vals[i-3]) / vals[i-3]
        decline[(yr, mo)] = max(-chg, 0.0)
    return decline


PALM_3MO_DECLINE = _palm_price_lookup()


# World Bank Pink Sheet monthly natural rubber price (RSS3 grade, USD/kg).
RUBBER_PRICE = {
    (2007, 6):2.32, (2007, 7):2.33, (2007, 8):2.13, (2007, 9):2.40, (2007,10):2.44,
    (2007,11):2.52, (2007,12):2.47,
    (2008, 1):2.68, (2008, 2):2.90, (2008, 3):3.09, (2008, 4):3.08, (2008, 5):3.15,
    (2008, 6):3.43, (2008, 7):3.18, (2008, 8):2.82, (2008, 9):2.50,
    (2014, 8):1.78, (2014, 9):1.67, (2014,10):1.53, (2014,11):1.52, (2014,12):1.54,
    (2015, 1):1.58, (2015, 2):1.59, (2015, 3):1.52, (2015, 4):1.52, (2015, 5):1.62,
    (2015, 6):1.61, (2015, 7):1.52, (2015, 8):1.38, (2015, 9):1.38, (2015,10):1.40,
    (2015,11):1.32, (2015,12):1.30,
}

# World Bank Pink Sheet monthly coffee Robusta price (cents/lb).
COFFEE_PRICE = {
    (2007, 6): 88.4, (2007, 7): 92.8, (2007, 8): 91.9, (2007, 9): 95.6, (2007,10): 93.5,
    (2007,11): 96.0, (2007,12): 99.4,
    (2008, 1):109.1, (2008, 2):110.4, (2008, 3):113.0, (2008, 4):116.4, (2008, 5):119.2,
    (2008, 6):116.0, (2008, 7):117.4, (2008, 8):112.0, (2008, 9):100.4,
    (2014, 8):100.5, (2014, 9): 99.2, (2014,10):102.4, (2014,11): 92.6, (2014,12): 90.0,
    (2015, 1): 88.4, (2015, 2): 86.9, (2015, 3): 80.0, (2015, 4): 78.1, (2015, 5): 80.5,
    (2015, 6): 80.6, (2015, 7): 80.5, (2015, 8): 78.3, (2015, 9): 76.9, (2015,10): 73.6,
    (2015,11): 70.3, (2015,12): 71.4,
}


def load_data(mode: str = "ifls5_only") -> pd.DataFrame:
    """mode: 'ifls5_only' or 'pooled_z'.
       'pooled_z' standardises cesd_raw within wave and adds wave FE in formulas."""
    df = pd.read_parquet(OUT / "analysis_dataset.parquet")
    fin = pd.read_parquet(OUT / "financial_shocks.parquet")
    finv2 = pd.read_parquet(OUT / "financial_shocks_v2.parquet")
    df["female"] = (df.sex == "F").astype(int)

    df = df.merge(fin, on=["pidlink", "wave"], how="left")
    df = df.merge(finv2, on=["pidlink", "wave"], how="left")
    hb_path = OUT / "health_bereavement_shocks.parquet"
    if hb_path.exists():
        df = df.merge(pd.read_parquet(hb_path), on=["pidlink", "wave"], how="left")
    fd_path = OUT / "finance_distress_shocks.parquet"
    if fd_path.exists():
        fd = pd.read_parquet(fd_path).drop(columns=["hhid"], errors="ignore")
        df = df.merge(fd, on=["pidlink", "wave"], how="left")

    if mode == "ifls5_only":
        df = df[df.wave == "IFLS5"].copy()
    elif mode == "ifls4_only":
        df = df[df.wave == "IFLS4"].copy()

    # Fill the new health/bereavement and financial-distress variables with 0 if missing
    for c in ["n_symptoms", "many_symptoms", "recent_hospitalised",
              "recent_accident_2y", "recently_widowed_5y",
              "debt_q4", "high_med_oop", "pce_decline_q4"]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    keep = ["cesd_raw", "tmean_c", "kab_code", "month", "year",
            "age", "sex", "edu_yrs", "married", "widowed",
            "job_loss_within_yr",
            "palm_farmer_individual", "rubber_farmer_individual", "coffee_farmer_individual",
            "palm_price_z", "interview_date", "wave"]
    df = df.dropna(subset=keep)

    # z-score CES-D within wave (collapses to single-wave z if mode='ifls5_only')
    df["cesd_z"] = df.groupby("wave")["cesd_raw"].transform(
        lambda s: (s - s.mean()) / s.std()
    )
    HEAT_MEAN = df.tmean_c.mean()
    HEAT_SD = df.tmean_c.std()
    df["heat_c"] = df.tmean_c - HEAT_MEAN
    df["_heat_mean"] = HEAT_MEAN
    df["_heat_sd"] = HEAT_SD
    df["_mode"] = mode

    df["post_subsidy"] = (df.interview_date >= pd.Timestamp("2014-11-18")).astype(int)
    df["intvw_yr_mo"] = list(zip(df.interview_date.dt.year, df.interview_date.dt.month))
    # Primary palm shock (3-month price decline × palm farmer):
    #   captures the "income just got hit" shock farmers actually experience, rather
    #   than a long-run-mean comparison. Identified in both waves where prices moved.
    df["palm_3mo_decline"] = df.intvw_yr_mo.map(PALM_3MO_DECLINE).fillna(0.0)
    df["palm_shock"] = df.palm_farmer_individual * df.palm_3mo_decline

    # Rubber: monthly RSS3 price → z-score → shock for rubber farmers when below mean
    df["rubber_price_usd_kg"] = df.intvw_yr_mo.map(RUBBER_PRICE)
    rp = pd.Series(list(RUBBER_PRICE.values()))
    df["rubber_price_z"] = (df.rubber_price_usd_kg - rp.mean()) / rp.std()
    df["rubber_shock"] = df.rubber_farmer_individual * (-df.rubber_price_z.fillna(0)).clip(lower=0)

    # Coffee: monthly Robusta price (cents/lb)
    df["coffee_price_clb"] = df.intvw_yr_mo.map(COFFEE_PRICE)
    cp = pd.Series(list(COFFEE_PRICE.values()))
    df["coffee_price_z"] = (df.coffee_price_clb - cp.mean()) / cp.std()
    df["coffee_shock"] = df.coffee_farmer_individual * (-df.coffee_price_z.fillna(0)).clip(lower=0)
    return df


def lincom(model, contrast: dict, name: str) -> dict:
    """Compute β = a'·coef and SE = sqrt(a' V a) for a linear combination.

    contrast: {coef_name: weight}
    """
    coefs = model.coef()
    vcov = pd.DataFrame(model._vcov, index=coefs.index, columns=coefs.index)
    a = pd.Series(0.0, index=coefs.index)
    for c, w in contrast.items():
        if c not in coefs.index:
            return {"name": name, "coef": np.nan, "se": np.nan, "t": np.nan, "p": np.nan}
        a[c] = w
    val = float(a @ coefs)
    se = float(np.sqrt(a @ vcov @ a))
    t = val / se if se > 0 else np.nan
    from scipy import stats
    p = 2 * (1 - stats.norm.cdf(abs(t)))
    return {"name": name, "coef": val, "se": se, "t": t, "p": p}


def run_one_model(df: pd.DataFrame, stressor: str, label: str,
                  scale_for_lincom: float = 1.0,
                  extra_controls: list[str] | None = None) -> dict:
    """Single specification: cesd_z ~ heat_c × stressor + controls + FE.

    Args:
      stressor          variable to interact with heat_c
      scale_for_lincom  multiplier when reporting "effect of 1-unit stressor"
                        (use 0.10 for transport_share -> 10pp jump)
      extra_controls    additional terms to include linearly without interaction.
                        Used to clean palm_shock by separating palm_farmer baseline.
    """
    fe = "month + year + kab_code"
    if df["_mode"].iloc[0] == "pooled_z":
        fe = "wave + " + fe
    extra = ""
    if extra_controls:
        extra = " + " + " + ".join(extra_controls)
    formula = (
        f"cesd_z ~ heat_c * {stressor} + age + female + edu_yrs + married + widowed{extra} "
        f"| {fe}"
    )
    m = pf.feols(formula, data=df, vcov={"CRV1": "kab_code"})

    # Headline coefficients
    main_heat = m.coef().get("heat_c", np.nan)
    se_heat = m.se().get("heat_c", np.nan)
    p_heat = m.pvalue().get("heat_c", np.nan)

    main_stress = m.coef().get(stressor, np.nan)
    se_stress = m.se().get(stressor, np.nan)
    p_stress = m.pvalue().get(stressor, np.nan)

    inter_term = f"heat_c:{stressor}"
    inter = m.coef().get(inter_term, np.nan)
    inter_se = m.se().get(inter_term, np.nan)
    inter_p = m.pvalue().get(inter_term, np.nan)

    # lincom marginal effects
    heat_sd = df["_heat_sd"].iloc[0]

    if scale_for_lincom == 1.0:
        # Binary stressor: report at stress=0 and stress=1
        eff_heat_at_stress0 = lincom(m, {"heat_c": 1.0}, "heat | stress=0")
        eff_heat_at_stress1 = lincom(m, {"heat_c": 1.0, inter_term: 1.0}, "heat | stress=1")
        eff_stress_at_heatM = lincom(m, {stressor: 1.0}, "stress | heat=mean")
        eff_stress_at_hot   = lincom(m, {stressor: 1.0, inter_term: heat_sd}, "stress | heat=+1SD")
    else:
        # Continuous stressor: report at stress=median and stress=high
        s = df[stressor]
        s_lo = s.quantile(0.50)
        s_hi = s.quantile(0.90)
        eff_heat_at_stress0 = lincom(m, {"heat_c": 1.0, inter_term: float(s_lo)}, "heat | stress=p50")
        eff_heat_at_stress1 = lincom(m, {"heat_c": 1.0, inter_term: float(s_hi)}, "heat | stress=p90")
        # for stressor effect we go in scale_for_lincom-units
        eff_stress_at_heatM = lincom(m, {stressor: scale_for_lincom},
                                     f"stress(+{scale_for_lincom}) | heat=mean")
        eff_stress_at_hot   = lincom(m, {stressor: scale_for_lincom, inter_term: scale_for_lincom * heat_sd},
                                     f"stress(+{scale_for_lincom}) | heat=+1SD")

    return {
        "stressor_label": label,
        "stressor_var":   stressor,
        "n":              int(m._N),
        "stress_share":   float(df[stressor].mean()),

        # Raw coefficients
        "heat_coef":      main_heat,
        "heat_se":        se_heat,
        "heat_p":         p_heat,
        "stress_coef":    main_stress,
        "stress_se":      se_stress,
        "stress_p":       p_stress,
        "inter_coef":     inter,
        "inter_se":       inter_se,
        "inter_p":        inter_p,

        # Marginal effects
        **{f"me_{k}_{v}": eff_heat_at_stress0[k] for k, v in {"name": "name", "coef": "coef",
                                                              "se": "se", "p": "p"}.items()
           if k in ["coef", "se", "p"]},
        "me_heat_lo_label": eff_heat_at_stress0["name"],
        "me_heat_lo_coef":  eff_heat_at_stress0["coef"],
        "me_heat_lo_se":    eff_heat_at_stress0["se"],
        "me_heat_lo_p":     eff_heat_at_stress0["p"],
        "me_heat_hi_label": eff_heat_at_stress1["name"],
        "me_heat_hi_coef":  eff_heat_at_stress1["coef"],
        "me_heat_hi_se":    eff_heat_at_stress1["se"],
        "me_heat_hi_p":     eff_heat_at_stress1["p"],
        "me_stress_at_mean_label": eff_stress_at_heatM["name"],
        "me_stress_at_mean_coef":  eff_stress_at_heatM["coef"],
        "me_stress_at_mean_se":    eff_stress_at_heatM["se"],
        "me_stress_at_mean_p":     eff_stress_at_heatM["p"],
        "me_stress_at_hot_label":  eff_stress_at_hot["name"],
        "me_stress_at_hot_coef":   eff_stress_at_hot["coef"],
        "me_stress_at_hot_se":     eff_stress_at_hot["se"],
        "me_stress_at_hot_p":      eff_stress_at_hot["p"],
    }


def run_all(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = []
    rows.append(run_one_model(df, "job_loss_within_yr", "Job loss within 12 mo (binary)"))
    # Palm shock with palm_farmer_individual as a separate control so the shock
    # coefficient isolates the price-collapse effect, not cross-sectional palm-farmer levels.
    rows.append(run_one_model(df, "palm_shock", "Palm-price shock × palm farmer",
                              scale_for_lincom=1.0,
                              extra_controls=["palm_farmer_individual"]))
    # Same template for rubber: rubber_farmer × max(−rubber_price_z, 0).
    rows.append(run_one_model(df, "rubber_shock", "Rubber-price shock × rubber farmer",
                              scale_for_lincom=1.0,
                              extra_controls=["rubber_farmer_individual"]))
    # Same template for coffee.
    rows.append(run_one_model(df, "coffee_shock", "Coffee-price shock × coffee farmer",
                              scale_for_lincom=1.0,
                              extra_controls=["coffee_farmer_individual"]))

    # Health & bereavement shocks (kept for the scope-condition discussion).
    if "many_symptoms" in df.columns:
        rows.append(run_one_model(df, "many_symptoms", "Many symptoms in past 4 wks (≥5)"))
    if "recent_hospitalised" in df.columns:
        rows.append(run_one_model(df, "recent_hospitalised", "Hospitalised in past 12 mo"))
    if "recent_accident_2y" in df.columns:
        rows.append(run_one_model(df, "recent_accident_2y", "Accident with treatment in past 2 yrs"))
    if "recently_widowed_5y" in df.columns:
        rows.append(run_one_model(df, "recently_widowed_5y", "Widowed within last 5 yrs"))

    # NEW financial-distress shocks
    if "debt_q4" in df.columns:
        rows.append(run_one_model(df, "debt_q4", "High HH debt (top quartile of borrowers)"))
    if "high_med_oop" in df.columns:
        rows.append(run_one_model(df, "high_med_oop", "Large medical OOP (top quartile of hospitalised)"))
    if "pce_decline_q4" in df.columns:
        rows.append(run_one_model(df, "pce_decline_q4", "Inter-wave PCE decline (bottom quartile, panel only)"))
    out = pd.DataFrame(rows)
    out["mode"] = tag
    return out


def main() -> None:
    all_results = []
    for mode, label in [("pooled_z", "Pooled IFLS4+5 (CES-D z within wave)"),
                        ("ifls4_only", "IFLS4 only"),
                        ("ifls5_only", "IFLS5 only")]:
        df = load_data(mode)
        HEAT_MEAN = df["_heat_mean"].iloc[0]
        HEAT_SD = df["_heat_sd"].iloc[0]
        print(f"\n{'='*70}\n{label}: n = {len(df):,}\n{'='*70}")
        print(f"  heat_c centered at = {HEAT_MEAN:.2f}°C, sd = {HEAT_SD:.2f}°C")
        print(f"  job_loss_within_yr share: {df.job_loss_within_yr.mean():.3f}")
        print(f"  palm_farmer_individual share: {df.palm_farmer_individual.mean():.3f}")
        print(f"  transport_share median: {df.transport_share.median():.3f}, p90: {df.transport_share.quantile(0.9):.3f}")

        out = run_all(df, label)
        cols_main = ["stressor_label", "n", "stress_share",
                     "heat_coef", "heat_p", "stress_coef", "stress_p",
                     "inter_coef", "inter_se", "inter_p"]
        print(f"\n--- Headline coefficients ---")
        print(out[cols_main].round(4).to_string(index=False))

        print(f"\n--- Lincom marginal effects ---")
        for _, r in out.iterrows():
            print(f"\n  ◆ {r['stressor_label']}")
            for tag in ["heat_lo", "heat_hi", "stress_at_mean", "stress_at_hot"]:
                lab = r[f"me_{tag}_label"]
                c, se, p = r[f"me_{tag}_coef"], r[f"me_{tag}_se"], r[f"me_{tag}_p"]
                star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
                print(f"      {lab:>34s}:  β = {c:+.4f}  ({se:.4f})  p={p:.4f} {star}")

        all_results.append(out)

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(RES / "table_unified_refined.csv", index=False)
    print(f"\nwrote {RES/'table_unified_refined.csv'}")

    # Side-by-side comparison of the headline interaction
    print("\n\n" + "="*70)
    print("SIDE-BY-SIDE: headline interaction β (cesd_z per °C × stressor)")
    print("="*70)
    pivot = final.pivot_table(
        index="stressor_label", columns="mode",
        values=["inter_coef", "inter_p", "n"], aggfunc="first",
    )
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
