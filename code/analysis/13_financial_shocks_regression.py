"""Heat × financial-shock interactions, full battery.

Tests:
  H. Heat × recent job loss (5y / 12mo / involuntary)
  I. Heat × post-Nov-2014 fuel-subsidy cut × vehicle/urban dependence  (DiD-style)
  J. Heat × palm region × palm-price shock  (commodity bust DiD)
  K. Heat × cash-transfer recipient (cushion / reverse-direction test)
  L. Combined ranking: heat × {all stressors air pollution + financial}
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "data" / "generated"
RES = OUT / "results"
RES.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(OUT / "analysis_dataset.parquet")
    fin = pd.read_parquet(OUT / "financial_shocks.parquet")
    aod = pd.read_parquet(OUT / "aod_monthly_kab.parquet")
    df["female"] = (df.sex == "F").astype(int)
    df["pce_q1"] = (df.pce_quintile == 1).astype(int)
    df["age_60p"] = (df.age >= 60).astype(int)

    df = df.merge(fin, on=["pidlink", "wave"], how="left")
    df["intvw_yr"] = df.interview_date.dt.year
    df["intvw_mo"] = df.interview_date.dt.month
    df = df.merge(aod.rename(columns={"year": "intvw_yr", "month": "intvw_mo"}),
                  on=["kabupaten_code", "intvw_yr", "intvw_mo"], how="left")

    keep = ["cesd_raw", "tmean_c", "tmax_c", "kabupaten_code", "wave", "month", "year",
            "age", "sex", "edu_yrs", "married", "widowed",
            "recent_job_loss_5y", "involuntary_loss_5y", "job_loss_within_yr",
            "vehicle_owner", "urban", "cash_transfer_recipient",
            "palm_region", "palm_price_z"]
    df = df.dropna(subset=keep)
    return df


def table_h_job_loss(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("Recent job loss 5y main",
         "cesd_raw ~ recent_job_loss_5y + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("Tmean × recent job loss 5y",
         "cesd_raw ~ tmean_c * recent_job_loss_5y + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("Tmean × involuntary loss",
         "cesd_raw ~ tmean_c * involuntary_loss_5y + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("Tmean × job loss within 12mo",
         "cesd_raw ~ tmean_c * job_loss_within_yr + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("Tmax × job loss within 12mo",
         "cesd_raw ~ tmax_c * job_loss_within_yr + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
    ]
    for label, formula in specs:
        try:
            m = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_code"})
            rec = {"spec": label, "n": int(m._N)}
            for t in m.coef().index:
                if any(k in t for k in ["job_loss", "involuntary", "tmean", "tmax"]):
                    rec[f"{t}_coef"] = m.coef()[t]
                    rec[f"{t}_p"] = m.pvalue()[t]
            rows.append(rec)
        except Exception as e:
            print(f"  failed {label}: {e}")
            rows.append({"spec": label, "n": 0})
    out = pd.DataFrame(rows)
    print("\n=== TABLE H: heat × job loss ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(RES / "tableH_job_loss.csv", index=False)
    return out


def table_i_fuel_subsidy(df: pd.DataFrame) -> pd.DataFrame:
    """Post-Nov-2014 fuel-subsidy cut interacted with HH transport dependence.
       Identification within IFLS5 only since that's where the cut falls."""
    sub = df[df.wave == "IFLS5"].copy()
    sub["post_subsidy"] = (sub.interview_date >= pd.Timestamp("2014-11-18")).astype(int)
    print(f"\n--- Subsidy DiD subsample (IFLS5 only): n={len(sub)} ---")
    print(f"  post-subsidy share: {sub.post_subsidy.mean():.3f}")
    print(f"  vehicle owner share: {sub.vehicle_owner.mean():.3f}")
    print(f"  urban share: {sub.urban.mean():.3f}")

    rows = []
    specs = [
        ("Subsidy: post × vehicle (DiD)",
         "cesd_raw ~ post_subsidy * vehicle_owner + age + female + edu_yrs + married + widowed | month + year + kabupaten_code"),
        ("Subsidy: post × urban (DiD)",
         "cesd_raw ~ post_subsidy * urban + age + female + edu_yrs + married + widowed | month + year + kabupaten_code"),
        ("Subsidy + heat (3-way: post × vehicle × tmean)",
         "cesd_raw ~ tmean_c * post_subsidy * vehicle_owner + age + female + edu_yrs + married + widowed | month + year + kabupaten_code"),
        ("Subsidy + heat (3-way: post × urban × tmean)",
         "cesd_raw ~ tmean_c * post_subsidy * urban + age + female + edu_yrs + married + widowed | month + year + kabupaten_code"),
    ]
    for label, formula in specs:
        try:
            m = pf.feols(formula, data=sub, vcov={"CRV1": "kabupaten_code"})
            rec = {"spec": label, "n": int(m._N)}
            for t in m.coef().index:
                if any(k in t for k in ["post_subsidy", "vehicle", "urban", "tmean"]):
                    rec[f"{t}_coef"] = m.coef()[t]
                    rec[f"{t}_p"] = m.pvalue()[t]
            rows.append(rec)
        except Exception as e:
            print(f"  failed {label}: {e}")
            rows.append({"spec": label, "n": 0})
    out = pd.DataFrame(rows)
    print("\n=== TABLE I: fuel-subsidy cut × dependence ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(RES / "tableI_fuel_subsidy.csv", index=False)
    return out


def table_j_palm_shock(df: pd.DataFrame) -> pd.DataFrame:
    """Heat × palm-region × palm-price shock. Continuous price × static region.
       palm_price_z is monthly z-score of the palm price; negative z = price collapse."""
    rows = []
    df_p = df.dropna(subset=["palm_price_z"]).copy()
    # Define palm shock as price LOWER than long-run mean (negative z)
    df_p["palm_shock"] = df_p.palm_region * (-df_p.palm_price_z).clip(lower=0)

    specs = [
        ("Palm shock main (no heat)",
         "cesd_raw ~ palm_shock + palm_region + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("Tmean × palm shock",
         "cesd_raw ~ tmean_c * palm_shock + palm_region + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("Tmean × palm region × palm price (3-way)",
         "cesd_raw ~ tmean_c * palm_region * palm_price_z + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
    ]
    for label, formula in specs:
        try:
            m = pf.feols(formula, data=df_p, vcov={"CRV1": "kabupaten_code"})
            rec = {"spec": label, "n": int(m._N)}
            for t in m.coef().index:
                if any(k in t for k in ["palm", "tmean"]):
                    rec[f"{t}_coef"] = m.coef()[t]
                    rec[f"{t}_p"] = m.pvalue()[t]
            rows.append(rec)
        except Exception as e:
            print(f"  failed {label}: {e}")
            rows.append({"spec": label, "n": 0})
    out = pd.DataFrame(rows)
    print("\n=== TABLE J: heat × palm-region × palm price ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(RES / "tableJ_palm_shock.csv", index=False)
    return out


def table_k_cushion(df: pd.DataFrame) -> pd.DataFrame:
    """Cash transfer / safety-net cushion test. Predicted sign: NEGATIVE interaction
       (recipients should be BUFFERED from heat-driven CES-D rises)."""
    rows = []
    specs = [
        ("Cash transfer × tmean",
         "cesd_raw ~ tmean_c * cash_transfer_recipient + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("Health card × tmean (broader safety net)",
         "cesd_raw ~ tmean_c * health_card + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("Cash transfer + AOD × tmean (3-way)",
         "cesd_raw ~ tmean_c * cash_transfer_recipient * aod + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
    ]
    df["health_card"] = df.get("health_card", 0)
    for label, formula in specs:
        try:
            data = df if "aod" not in formula else df.dropna(subset=["aod"])
            m = pf.feols(formula, data=data, vcov={"CRV1": "kabupaten_code"})
            rec = {"spec": label, "n": int(m._N)}
            for t in m.coef().index:
                if any(k in t for k in ["cash_transfer", "health_card", "tmean", "aod"]):
                    rec[f"{t}_coef"] = m.coef()[t]
                    rec[f"{t}_p"] = m.pvalue()[t]
            rows.append(rec)
        except Exception as e:
            print(f"  failed {label}: {e}")
            rows.append({"spec": label, "n": 0})
    out = pd.DataFrame(rows)
    print("\n=== TABLE K: cushion test (cash transfer / health card × heat) ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(RES / "tableK_cushion.csv", index=False)
    return out


def table_l_horse_race(df: pd.DataFrame) -> pd.DataFrame:
    """Final ranking: each stressor's heat-amplification effect head-to-head.

    For each stressor S, run cesd_raw ~ tmean_c * S + controls + FE.
    Sort by interaction p-value. The most amplifying stressor wins.
    """
    candidates = [
        ("recent_job_loss_5y", "Job loss in past 5y"),
        ("involuntary_loss_5y", "Involuntary loss"),
        ("job_loss_within_yr", "Job loss within 12mo"),
        ("vehicle_owner", "Vehicle owner"),
        ("urban", "Urban resident"),
        ("cash_transfer_recipient", "Cash transfer recipient"),
        ("health_card", "Health card / safety net"),
        ("palm_region", "Palm-oil region"),
        ("widowed", "Widowed"),
        ("pce_q1", "Bottom PCE quintile"),
        ("age_60p", "Age 60+"),
        ("female", "Female"),
    ]
    rows = []
    for var, label in candidates:
        if var not in df.columns:
            continue
        formula = (f"cesd_raw ~ tmean_c * {var} + age + female + edu_yrs + married + widowed "
                   f"| wave + month + year + kabupaten_code")
        try:
            m = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_code"})
            inter = f"tmean_c:{var}"
            rows.append({
                "stressor": label,
                "var": var,
                "stress_share": float(df[var].mean()),
                "tmean_main_coef": m.coef().get("tmean_c", np.nan),
                "tmean_main_p":    m.pvalue().get("tmean_c", np.nan),
                "stress_main_coef": m.coef().get(var, np.nan),
                "stress_main_p":    m.pvalue().get(var, np.nan),
                "interaction_coef": m.coef().get(inter, np.nan),
                "interaction_se":   m.se().get(inter, np.nan),
                "interaction_t":    m.tstat().get(inter, np.nan),
                "interaction_p":    m.pvalue().get(inter, np.nan),
                "n": int(m._N),
            })
        except Exception as e:
            print(f"  failed {var}: {e}")
            rows.append({"stressor": label, "var": var, "interaction_p": np.nan, "n": 0})
    out = pd.DataFrame(rows).sort_values("interaction_p")
    print("\n=== TABLE L: HORSE RACE (heat × each stressor, ranked by p) ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(RES / "tableL_horse_race.csv", index=False)
    return out


def main() -> None:
    df = load_data()
    print(f"sample: {len(df):,}")

    table_h_job_loss(df)
    table_i_fuel_subsidy(df)
    table_j_palm_shock(df)
    table_k_cushion(df)
    table_l_horse_race(df)


if __name__ == "__main__":
    main()
