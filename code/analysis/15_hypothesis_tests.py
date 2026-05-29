"""Hypothesis decomposition: somatic-specificity, day/night, applied to job loss AND palm shock.

For each outcome (CES-D total z; Somatic z; Depressed-Affect z; Positive-Affect z)
and each heat measure (tmean_c, tmax_c, tmin_c) and each stressor
(job_loss_within_yr; palm_shock), we run

    factor_z ~ heat_c * stressor + controls + month + year + wave + kabupaten_code FE

clustered by kabupaten_code. We report the heat * stressor interaction with significance,
plus the placebo combinations expected to be null.

Radloff factor mapping for the 10-item CES-D used in IFLS:
  Somatic         : A bothered, B trouble keeping mind, D everything an effort,
                    G restless sleep, J could not get going          (5 items)
  Depressed Affect: C depressed, F fearful, I lonely                 (3 items)
  Positive Affect : E hopeful (reverse), H happy (reverse)           (2 items)
  (Interpersonal subscale absent in 10-item version)

Score-construction parallels score_cesd.py:
  IFLS5: 0..3 per item from kp02 - 1, reverse for E/H, sum within factor
  IFLS4: 0..3 per item from kp02 - 1 when kp01==1 (else 0); 1.5 if kp01=1 & kp02 NaN
         (matches the cesd_raw construction; same wave-internal scale)
Then we z-score each factor within wave so coefficients read in SDs.

Output: data/generated/results/table_hypothesis_tests.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

from _table_input import load_table_input

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "data" / "generated"
RES = OUT / "results"
RES.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = load_table_input()
    if df[["somatic_z", "depraffect_z", "posaffect_z"]].isna().all().all():
        raise RuntimeError(
            "analysis_table_input.parquet does not contain CES-D factor scores. "
            "Mount RAW IFLS item files and rerun code/data/13_build_analysis_table_input.py."
        )

    keep = ["tmean_c", "tmax_c", "tmin_c", "kabupaten_code", "month", "year", "wave",
            "age", "edu_yrs", "married", "widowed", "female", "interview_date",
            "job_loss_within_yr", "palm_farmer_individual", "transport_share",
            "somatic_z", "depraffect_z", "posaffect_z"]
    df = df.dropna(subset=keep)

    # Mean-center each heat measure on the SAMPLE mean (after dropna) so interactions
    # interpret cleanly.
    for h in ["tmean_c", "tmax_c", "tmin_c"]:
        df[f"{h}_dev"] = df[h] - df[h].mean()
    return df


# ---------------- Model harness ------------------------------------------------------------

def run_inter(df: pd.DataFrame, outcome: str, heat: str, stressor: str,
              extra_controls: list[str] | None = None) -> dict:
    fe = "wave + month + year + kabupaten_code"
    extra = ""
    if extra_controls:
        extra = " + " + " + ".join(extra_controls)
    formula = (
        f"{outcome} ~ {heat} * {stressor} + age + female + edu_yrs + married + widowed{extra} "
        f"| {fe}"
    )
    m = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_code"})
    inter = f"{heat}:{stressor}"
    coef = m.coef().get(inter, np.nan)
    se = m.se().get(inter, np.nan)
    p = m.pvalue().get(inter, np.nan)
    heat_b = m.coef().get(heat, np.nan)
    heat_p = m.pvalue().get(heat, np.nan)
    return dict(outcome=outcome, heat=heat, stressor=stressor,
                inter_coef=coef, inter_se=se, inter_p=p,
                heat_coef=heat_b, heat_p=heat_p, n=int(m._N))


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def main() -> None:
    df = load_data()
    print(f"loaded n = {len(df):,}; waves = {df.wave.value_counts().to_dict()}")
    print(f"  palm_shock>0 share: {(df.palm_shock>0).mean():.4f}")
    print(f"  job_loss share:     {df.job_loss_within_yr.mean():.4f}")
    print(f"  somatic items mean: {df.somatic_z.mean():.2f}  sd: {df.somatic_z.std():.2f}")

    rows = []
    grid = [
        # (heat, stressor, extra_controls)
        ("tmean_c_dev", "job_loss_within_yr", None),
        ("tmax_c_dev",  "job_loss_within_yr", None),
        ("tmin_c_dev",  "job_loss_within_yr", None),
        ("tmean_c_dev", "palm_shock", ["palm_farmer_individual"]),
        ("tmax_c_dev",  "palm_shock", ["palm_farmer_individual"]),
        ("tmin_c_dev",  "palm_shock", ["palm_farmer_individual"]),
    ]
    outcomes = ["cesd_z", "somatic_z", "depraffect_z", "posaffect_z"]
    for heat, stressor, extra in grid:
        for outcome in outcomes:
            try:
                r = run_inter(df, outcome, heat, stressor, extra_controls=extra)
                rows.append(r)
            except Exception as e:
                print(f"  fail {outcome} ~ {heat}*{stressor}: {e}")

    res = pd.DataFrame(rows)
    res["stars"] = res.inter_p.map(stars)
    res.to_csv(RES / "table_hypothesis_tests.csv", index=False)

    # Pivot for compact display: rows = stressor x heat, cols = outcome
    res["coef_str"] = res.apply(
        lambda r: f"{r.inter_coef:+.4f}{r.stars} (p={r.inter_p:.3f})", axis=1)
    pivot = res.pivot_table(index=["stressor", "heat"], columns="outcome",
                            values="coef_str", aggfunc="first")
    pivot = pivot[outcomes]
    print("\n=" * 0 + "=" * 90)
    print("INTERACTION COEFFICIENTS  (heat × stressor → outcome z-score)")
    print("=" * 90)
    print(pivot.to_string())
    print(f"\nwrote {RES/'table_hypothesis_tests.csv'}")

    # Also direct heat effects (no interaction) for completeness
    print("\n" + "=" * 90)
    print("DIRECT HEAT EFFECTS (no stressor interaction)")
    print("=" * 90)
    direct = []
    for outcome in outcomes:
        for heat in ["tmean_c_dev", "tmax_c_dev", "tmin_c_dev"]:
            f = (f"{outcome} ~ {heat} + age + female + edu_yrs + married + widowed "
                 f"| wave + month + year + kabupaten_code")
            m = pf.feols(f, data=df, vcov={"CRV1": "kabupaten_code"})
            direct.append(dict(outcome=outcome, heat=heat,
                               coef=m.coef().get(heat, np.nan),
                               se=m.se().get(heat, np.nan),
                               p=m.pvalue().get(heat, np.nan)))
    dres = pd.DataFrame(direct)
    dres["stars"] = dres.p.map(stars)
    dres["coef_str"] = dres.apply(lambda r: f"{r.coef:+.4f}{r.stars} (p={r.p:.3f})", axis=1)
    print(dres.pivot_table(index="heat", columns="outcome", values="coef_str", aggfunc="first")[outcomes].to_string())
    dres.to_csv(RES / "table_direct_heat_effects.csv", index=False)

    # ---- Fuel-subsidy 3-way DiD on each factor (IFLS5 only) ----
    print("\n" + "=" * 90)
    print("FUEL-SUBSIDY SHOCK  (IFLS5 only — post_subsidy × transport_share)")
    print("=" * 90)
    sub = df[df.wave == "IFLS5"].copy()
    print(f"IFLS5 n = {len(sub):,};  post_subsidy share = {sub.post_subsidy.mean():.3f};"
          f"  transport_share median = {sub.transport_share.median():.3f}")
    # Re-center heat on the IFLS5 sub-sample mean so the interaction reads cleanly
    for h in ["tmean_c", "tmax_c", "tmin_c"]:
        sub[f"{h}_dev"] = sub[h] - sub[h].mean()
    sub["fuel_shock"] = sub.post_subsidy * sub.transport_share

    fuel_rows = []
    for outcome in outcomes:
        for heat in ["tmean_c_dev", "tmax_c_dev", "tmin_c_dev"]:
            try:
                # Spec: factor ~ heat * fuel_shock + transport_share + ...
                #   - month/year FE absorb post_subsidy's time-step
                #   - transport_share controls for cross-sectional level
                #   - heat * fuel_shock is the 3-way DiD (heat × post × transport-share)
                f = (f"{outcome} ~ {heat} * fuel_shock + transport_share + "
                     f"age + female + edu_yrs + married + widowed "
                     f"| month + year + kabupaten_code")
                m = pf.feols(f, data=sub, vcov={"CRV1": "kabupaten_code"})
                inter = f"{heat}:fuel_shock"
                fuel_rows.append(dict(
                    outcome=outcome, heat=heat,
                    inter_coef=m.coef().get(inter, np.nan),
                    inter_se=m.se().get(inter, np.nan),
                    inter_p=m.pvalue().get(inter, np.nan),
                    fuel_coef=m.coef().get("fuel_shock", np.nan),
                    fuel_p=m.pvalue().get("fuel_shock", np.nan),
                    n=int(m._N)))
            except Exception as e:
                print(f"  fail {outcome} ~ {heat}*fuel_shock: {e}")
    fr = pd.DataFrame(fuel_rows)
    fr["stars"] = fr.inter_p.map(stars)
    fr["coef_str"] = fr.apply(lambda r: f"{r.inter_coef:+.4f}{r.stars} (p={r.inter_p:.3f})", axis=1)
    print("\n  Heat × fuel_shock interaction (where fuel_shock = post_subsidy × transport_share):")
    print(fr.pivot_table(index="heat", columns="outcome", values="coef_str", aggfunc="first")[outcomes].to_string())
    fr["fuel_stars"] = fr.fuel_p.map(stars)
    fr["fuel_str"] = fr.apply(lambda r: f"{r.fuel_coef:+.4f}{r.fuel_stars} (p={r.fuel_p:.3f})", axis=1)
    print("\n  Post × transport_share (post-cut effect for high-transport HH, at heat = mean):")
    print(fr.pivot_table(index="heat", columns="outcome", values="fuel_str", aggfunc="first")[outcomes].to_string())
    fr.to_csv(RES / "table_fuel_hypothesis_tests.csv", index=False)
    print(f"\nwrote {RES/'table_fuel_hypothesis_tests.csv'}")

    # ---- Individual-FE robustness: within-person identification across IFLS4 → IFLS5 ----
    print("\n" + "=" * 90)
    print("INDIVIDUAL-FE ROBUSTNESS  (panel respondents, pidlink FE absorbs all time-invariant)")
    print("=" * 90)
    panel_pids = set(df[df.wave == "IFLS4"].pidlink) & set(df[df.wave == "IFLS5"].pidlink)
    panel = df[df.pidlink.isin(panel_pids)].copy()
    print(f"  panel respondents (in both waves): {len(panel_pids):,}")
    print(f"  panel-wave observations:           {len(panel):,}")
    print(f"  IFLS4 panel rows: {(panel.wave=='IFLS4').sum():,};  IFLS5 panel rows: {(panel.wave=='IFLS5').sum():,}")

    # Re-center heat on the panel sub-sample mean
    for h in ["tmean_c", "tmax_c", "tmin_c"]:
        panel[f"{h}_dev"] = panel[h] - panel[h].mean()
    panel["fuel_shock"] = panel.post_subsidy * panel.transport_share

    # Check identifying variation: how many panel respondents have within-person change?
    print("\n  Within-person variation in stressors (share of panel respondents with change):")
    for s in ["job_loss_within_yr", "palm_shock", "fuel_shock"]:
        chg = panel.groupby("pidlink")[s].nunique()
        share = (chg > 1).mean()
        print(f"    {s:>22s}:  {share:.3f} of panel respondents have ≥2 distinct values")

    panel_rows = []
    panel_grid = [
        ("tmean_c_dev", "job_loss_within_yr", None),
        ("tmax_c_dev",  "job_loss_within_yr", None),
        ("tmin_c_dev",  "job_loss_within_yr", None),
        ("tmean_c_dev", "palm_shock", ["palm_farmer_individual"]),
        ("tmax_c_dev",  "palm_shock", ["palm_farmer_individual"]),
        ("tmin_c_dev",  "palm_shock", ["palm_farmer_individual"]),
        ("tmean_c_dev", "fuel_shock", ["transport_share"]),
        ("tmax_c_dev",  "fuel_shock", ["transport_share"]),
        ("tmin_c_dev",  "fuel_shock", ["transport_share"]),
    ]
    for heat, stressor, extra in panel_grid:
        for outcome in outcomes:
            try:
                extra_str = ""
                if extra:
                    extra_str = " + " + " + ".join(extra)
                # FE: pidlink absorbs all time-invariant person attributes.
                # Keep month/year/kabupaten_code FE for general trend + local-shock absorption.
                f = (f"{outcome} ~ {heat} * {stressor} + age + edu_yrs + married + widowed{extra_str} "
                     f"| pidlink + month + year + kabupaten_code")
                m = pf.feols(f, data=panel, vcov={"CRV1": "kabupaten_code"})
                inter = f"{heat}:{stressor}"
                panel_rows.append(dict(
                    outcome=outcome, heat=heat, stressor=stressor,
                    inter_coef=m.coef().get(inter, np.nan),
                    inter_se=m.se().get(inter, np.nan),
                    inter_p=m.pvalue().get(inter, np.nan),
                    n=int(m._N)))
            except Exception as e:
                print(f"  fail {outcome} ~ {heat}*{stressor}: {e}")

    pres = pd.DataFrame(panel_rows)
    pres["stars"] = pres.inter_p.map(stars)
    pres["coef_str"] = pres.apply(lambda r: f"{r.inter_coef:+.4f}{r.stars} (p={r.inter_p:.3f})", axis=1)
    pres.to_csv(RES / "table_individual_fe_panel.csv", index=False)
    print(f"\n  Heat × stressor interaction with individual FE  (n_panel-waves = {len(panel):,}):")
    pivot = pres.pivot_table(index=["stressor", "heat"], columns="outcome",
                             values="coef_str", aggfunc="first")[outcomes]
    print(pivot.to_string())
    print(f"\nwrote {RES/'table_individual_fe_panel.csv'}")


if __name__ == "__main__":
    main()
