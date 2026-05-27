"""Hypothesis decomposition: somatic-specificity, day/night, applied to job loss AND palm shock.

For each outcome (CES-D total z; Somatic z; Depressed-Affect z; Positive-Affect z)
and each heat measure (tmean_c, tmax_c, tmin_c) and each stressor
(job_loss_within_yr; palm_shock), we run

    factor_z ~ heat_c * stressor + controls + month + year + wave + kab_code FE

clustered by kab_code. We report the heat * stressor interaction with significance,
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

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[2]
RAW = Path("E:/IFLS/extracted")
OUT = PROJECT / "data" / "generated"
RES = OUT / "results"
RES.mkdir(parents=True, exist_ok=True)

REVERSE_ITEMS = {"E", "H"}
FACTOR_MAP = {
    "somatic":    {"A", "B", "D", "G", "J"},
    "depraffect": {"C", "F", "I"},
    "posaffect":  {"E", "H"},
}


def _score_item(df: pd.DataFrame, wave: str) -> pd.DataFrame:
    """Return long df: pidlink, kptype, score (0..3 with E,H reverse coded)."""
    if wave == "IFLS5":
        df = df[df.kp02.between(1, 4)].copy()
        df["score"] = df.kp02 - 1
    else:  # IFLS4 screener
        df = df[df.kp01.isin([1, 3])].copy()
        df["score"] = 0.0
        has_freq = df.kp01.eq(1) & df.kp02.between(1, 4)
        df.loc[has_freq, "score"] = df.loc[has_freq, "kp02"] - 1
        df.loc[df.kp01.eq(1) & df.kp02.isna(), "score"] = 1.5
    rev = df.kptype.isin(REVERSE_ITEMS)
    df.loc[rev, "score"] = 3 - df.loc[rev, "score"]
    return df[["pidlink", "kptype", "score"]]


def build_factor_scores() -> pd.DataFrame:
    items5 = _score_item(pd.read_stata(RAW / "IFLS5/hh14/b3b_kp.dta", convert_categoricals=False), "IFLS5")
    items4 = _score_item(pd.read_stata(RAW / "IFLS4/hh07/b3b_kp.dta", convert_categoricals=False), "IFLS4")
    items5["wave"] = "IFLS5"; items4["wave"] = "IFLS4"
    items = pd.concat([items4, items5], ignore_index=True)

    out_rows = []
    for factor_name, items_set in FACTOR_MAP.items():
        sub = items[items.kptype.isin(items_set)]
        s = sub.groupby(["pidlink", "wave"])["score"].sum().rename(factor_name).reset_index()
        out_rows.append(s)
    out = out_rows[0]
    for piece in out_rows[1:]:
        out = out.merge(piece, on=["pidlink", "wave"], how="outer")

    for f in FACTOR_MAP:
        out[f"{f}_z"] = out.groupby("wave")[f].transform(lambda x: (x - x.mean()) / x.std())
    return out


# ---------------- Pull the analysis dataset & build stressors (mirrors 14_unified_refined) -----

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


def _palm_decline_lookup() -> dict:
    rows = sorted(PALM_PRICE_FULL.items())
    keys = [k for k, _ in rows]; vals = [v for _, v in rows]
    out = {}
    for i, (yr, mo) in enumerate(keys):
        if i < 3:
            continue
        py, pm = keys[i - 3]
        gap = (yr - py) * 12 + (mo - pm)
        if gap != 3:
            continue
        chg = (vals[i] - vals[i - 3]) / vals[i - 3]
        out[(yr, mo)] = max(-chg, 0.0)
    return out


PALM_3MO_DECLINE = _palm_decline_lookup()


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(OUT / "analysis_dataset.parquet")
    fin = pd.read_parquet(OUT / "financial_shocks.parquet")
    finv2 = pd.read_parquet(OUT / "financial_shocks_v2.parquet")
    df = df.merge(fin, on=["pidlink", "wave"], how="left")
    df = df.merge(finv2, on=["pidlink", "wave"], how="left")
    df["female"] = (df.sex == "F").astype(int)

    factor = build_factor_scores()
    df = df.merge(factor, on=["pidlink", "wave"], how="left")

    keep = ["tmean_c", "tmax_c", "tmin_c", "kab_code", "month", "year", "wave",
            "age", "edu_yrs", "married", "widowed", "female", "interview_date",
            "job_loss_within_yr", "palm_farmer_individual", "transport_share",
            "somatic_z", "depraffect_z", "posaffect_z"]
    df = df.dropna(subset=keep)

    # CES-D total z (wave-z) for completeness
    df["cesd_z"] = df.groupby("wave")["cesd_raw"].transform(lambda x: (x - x.mean()) / x.std())

    # Mean-center each heat measure on the SAMPLE mean (after dropna) so interactions
    # interpret cleanly.
    for h in ["tmean_c", "tmax_c", "tmin_c"]:
        df[f"{h}_dev"] = df[h] - df[h].mean()

    df["intvw_yr_mo"] = list(zip(df.interview_date.dt.year, df.interview_date.dt.month))
    df["palm_3mo_decline"] = df.intvw_yr_mo.map(PALM_3MO_DECLINE).fillna(0.0)
    df["palm_shock"] = df.palm_farmer_individual * df.palm_3mo_decline

    # Fuel-subsidy shock: post-Nov-2014 cut × transport-spending share.
    # IFLS4 has post_subsidy=0 throughout so fuel_shock is uniformly 0 in IFLS4;
    # the spec is identified within IFLS5 only and we run it that way below.
    df["post_subsidy"] = (df.interview_date >= pd.Timestamp("2014-11-18")).astype(int)
    df["fuel_shock"] = df.post_subsidy * df.transport_share
    return df


# ---------------- Model harness ------------------------------------------------------------

def run_inter(df: pd.DataFrame, outcome: str, heat: str, stressor: str,
              extra_controls: list[str] | None = None) -> dict:
    fe = "wave + month + year + kab_code"
    extra = ""
    if extra_controls:
        extra = " + " + " + ".join(extra_controls)
    formula = (
        f"{outcome} ~ {heat} * {stressor} + age + female + edu_yrs + married + widowed{extra} "
        f"| {fe}"
    )
    m = pf.feols(formula, data=df, vcov={"CRV1": "kab_code"})
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
                 f"| wave + month + year + kab_code")
            m = pf.feols(f, data=df, vcov={"CRV1": "kab_code"})
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
                     f"| month + year + kab_code")
                m = pf.feols(f, data=sub, vcov={"CRV1": "kab_code"})
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
                # Keep month/year/kab_code FE for general trend + local-shock absorption.
                f = (f"{outcome} ~ {heat} * {stressor} + age + edu_yrs + married + widowed{extra_str} "
                     f"| pidlink + month + year + kab_code")
                m = pf.feols(f, data=panel, vcov={"CRV1": "kab_code"})
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
