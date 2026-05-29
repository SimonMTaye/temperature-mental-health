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

from _table_input import load_table_input

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "data" / "generated"
RES = OUT / "results"
RES.mkdir(parents=True, exist_ok=True)


def load_data(mode: str = "ifls5_only") -> pd.DataFrame:
    """mode: 'ifls5_only' or 'pooled_z'.
       'pooled_z' standardises cesd_raw within wave and adds wave FE in formulas."""
    df = load_table_input()

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

    keep = ["cesd_raw", "tmean_c", "kabupaten_code", "month", "year",
            "age", "sex", "edu_yrs", "married", "widowed",
            "job_loss_within_yr",
            "palm_farmer_individual", "rubber_farmer_individual", "coffee_farmer_individual",
            "palm_shock", "rubber_shock", "coffee_shock", "interview_date", "wave"]
    df = df.dropna(subset=keep)
    counts = df.kabupaten_code.value_counts()
    df = df[df.kabupaten_code.isin(counts[counts > 1].index)].copy()

    HEAT_MEAN = df.tmean_c.mean()
    HEAT_SD = df.tmean_c.std()
    df["heat_c"] = df.tmean_c - HEAT_MEAN
    df["_heat_mean"] = HEAT_MEAN
    df["_heat_sd"] = HEAT_SD
    df["_mode"] = mode
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
    fe = "month + year + kabupaten_code"
    if df["_mode"].iloc[0] == "pooled_z":
        fe = "wave + " + fe
    extra = ""
    if extra_controls:
        extra = " + " + " + ".join(extra_controls)
    formula = (
        f"cesd_z ~ heat_c * {stressor} + age + female + edu_yrs + married + widowed{extra} "
        f"| {fe}"
    )
    m = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_code"})

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
