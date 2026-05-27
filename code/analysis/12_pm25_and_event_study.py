"""Three things:
  E. Daily PM2.5 from MERRA-2 — replaces monthly AOD with a direct daily pollution measure.
     Re-runs heat × PM2.5 in the pooled sample and in the haze subsample.
  F. Event study around 2015 haze onset.
     Pre = Jul-Aug 2015, Peak = Sep 2015 (IFLS5 fielding tapers after Sep so events later are weak).
     DiD: high-haze regions (Sumatra+Kalimantan) vs low-haze (Java+Bali+other), pre vs peak.
  G. Heat × PM2.5 × stressor 3-way (replaces AOD-version Table D).
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
    pm = pd.read_parquet(OUT / "pm25_daily_kab.parquet")
    pm["date"] = pd.to_datetime(pm.date)

    df["female"] = (df.sex == "F").astype(int)
    df["pce_q1"] = (df.pce_quintile == 1).astype(int)
    df["age_60p"] = (df.age >= 60).astype(int)

    # Match daily PM2.5 to interview date
    df = df.merge(
        pm[["kab_code", "date", "pm25_ugm3"]],
        left_on=["kab_code", "interview_date"], right_on=["kab_code", "date"],
        how="left",
    ).drop(columns=["date"])
    print(f"PM2.5 coverage: {df.pm25_ugm3.notna().sum()} / {len(df)}")

    # 7-day rolling mean PM2.5 per kab (chronic exposure)
    pm = pm.sort_values(["kab_code", "date"]).reset_index(drop=True)
    pm["pm25_7d"] = pm.groupby("kab_code")["pm25_ugm3"].rolling(7, min_periods=3).mean().reset_index(level=0, drop=True)
    pm["pm25_30d"] = pm.groupby("kab_code")["pm25_ugm3"].rolling(30, min_periods=15).mean().reset_index(level=0, drop=True)
    df = df.merge(
        pm[["kab_code", "date", "pm25_7d", "pm25_30d"]],
        left_on=["kab_code", "interview_date"], right_on=["kab_code", "date"],
        how="left",
    ).drop(columns=["date"])

    keep = ["cesd_raw", "tmean_c", "tmax_c", "kab_code", "wave", "month", "year",
            "age", "sex", "edu_yrs", "married", "widowed",
            "disaster_severe_5yr", "loan_rejected", "agri_occupation", "pm25_ugm3"]
    df = df.dropna(subset=keep)
    return df


def table_e_pm25(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- PM2.5 distribution ---")
    print(df.pm25_ugm3.describe().round(2))
    print(f"  IFLS4 mean: {df[df.wave=='IFLS4'].pm25_ugm3.mean():.1f} µg/m³")
    print(f"  IFLS5 mean: {df[df.wave=='IFLS5'].pm25_ugm3.mean():.1f} µg/m³")

    rows = []
    specs = [
        ("PM2.5 main effect (today)",
         "cesd_raw ~ pm25_ugm3 + age + female + edu_yrs + married + widowed | wave + month + year + kab_code"),
        ("PM2.5 7d main effect",
         "cesd_raw ~ pm25_7d + age + female + edu_yrs + married + widowed | wave + month + year + kab_code"),
        ("PM2.5 30d main effect",
         "cesd_raw ~ pm25_30d + age + female + edu_yrs + married + widowed | wave + month + year + kab_code"),
        ("Tmean × PM2.5 today",
         "cesd_raw ~ tmean_c * pm25_ugm3 + age + female + edu_yrs + married + widowed | wave + month + year + kab_code"),
        ("Tmean × PM2.5 7d",
         "cesd_raw ~ tmean_c * pm25_7d + age + female + edu_yrs + married + widowed | wave + month + year + kab_code"),
        ("Tmax × PM2.5 7d",
         "cesd_raw ~ tmax_c * pm25_7d + age + female + edu_yrs + married + widowed | wave + month + year + kab_code"),
        ("Tmean × log(PM2.5 7d)",
         "cesd_raw ~ tmean_c * I(np.log(pm25_7d)) + age + female + edu_yrs + married + widowed | wave + month + year + kab_code"),
    ]
    for label, formula in specs:
        try:
            m = pf.feols(formula, data=df, vcov={"CRV1": "kab_code"})
            rec = {"spec": label, "n": int(m._N)}
            for t in m.coef().index:
                if any(k in t for k in ["pm25", "tmean_c", "tmax_c"]):
                    rec[f"{t}_coef"] = m.coef()[t]
                    rec[f"{t}_p"] = m.pvalue()[t]
            rows.append(rec)
        except Exception as e:
            print(f"  failed {label}: {e}")
            rows.append({"spec": label, "n": 0})
    out = pd.DataFrame(rows)
    print("\n=== TABLE E: PM2.5 + heat × PM2.5 ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(RES / "tableE_pm25.csv", index=False)
    return out


def table_f_event_study(df: pd.DataFrame) -> pd.DataFrame:
    """DiD around 2015 haze onset.

    Treated = Sumatra (11-21) + Kalimantan (61-64).
    Control = Java (31-36) + Bali (51) + other.
    Pre = interviewed Jul or Aug 2015. Peak = Sep 2015 (Oct-Dec basically empty).
    """
    HAZE_PROVS = list(range(11, 22)) + list(range(61, 65))
    sub = df[df.wave == "IFLS5"].copy()
    sub["treated_region"] = sub.prov_code.isin(HAZE_PROVS).astype(int)

    sub["ym"] = sub.interview_date.dt.to_period("M").astype(str)
    sub_did = sub[sub.ym.isin(["2015-07", "2015-08", "2015-09"])].copy()
    sub_did["post"] = (sub_did.ym == "2015-09").astype(int)

    print(f"\n--- DiD subsample (IFLS5 Jul-Sep 2015): n={len(sub_did)} ---")
    print(sub_did.groupby(["treated_region", "post"]).size().unstack(fill_value=0))
    print("CES-D means:")
    print(sub_did.groupby(["treated_region", "post"]).cesd_raw.mean().unstack().round(2))
    print("PM2.5 means (μg/m³):")
    print(sub_did.groupby(["treated_region", "post"]).pm25_ugm3.mean().unstack().round(1))

    rows = []
    specs = [
        ("Basic DiD: treated × post",
         "cesd_raw ~ treated_region * post + age + female + edu_yrs + married + widowed | kab_code"),
        ("DiD + heat",
         "cesd_raw ~ treated_region * post + tmean_c + age + female + edu_yrs + married + widowed | kab_code"),
        ("DiD with continuous PM2.5",
         "cesd_raw ~ pm25_ugm3 + tmean_c + tmean_c:pm25_ugm3 + age + female + edu_yrs + married + widowed | kab_code"),
    ]
    for label, formula in specs:
        try:
            m = pf.feols(formula, data=sub_did, vcov={"CRV1": "kab_code"})
            rec = {"spec": label, "n": int(m._N)}
            for t in m.coef().index:
                if any(k in t for k in ["treated_region", "post", "pm25", "tmean", "kab_code:"]):
                    rec[f"{t}_coef"] = m.coef()[t]
                    rec[f"{t}_p"] = m.pvalue()[t]
            rows.append(rec)
        except Exception as e:
            print(f"  failed {label}: {e}")
            rows.append({"spec": label, "n": 0})
    out = pd.DataFrame(rows)
    print("\n=== TABLE F: 2015 haze onset DiD ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(RES / "tableF_haze_did.csv", index=False)
    return out


def table_g_three_way_pm25(df: pd.DataFrame) -> pd.DataFrame:
    """Heat × PM2.5 × stressor 3-way using daily PM2.5."""
    rows = []
    stressors = [
        ("pce_q1", "Bottom expenditure quintile"),
        ("disaster_severe_5yr", "Severe disaster in 5y"),
        ("agri_occupation", "Agriculture occupation"),
        ("widowed", "Widowed/separated"),
        ("age_60p", "Age 60+"),
        ("loan_rejected", "Loan rejected"),
    ]
    for var, label in stressors:
        formula = f"cesd_raw ~ tmean_c * pm25_7d * {var} + age + female + edu_yrs + married + widowed | wave + month + year + kab_code"
        try:
            m = pf.feols(formula, data=df, vcov={"CRV1": "kab_code"})
            three_way = f"tmean_c:pm25_7d:{var}"
            rows.append({
                "stressor": label,
                "var": var,
                "tmean_main": m.coef().get("tmean_c", np.nan),
                "pm25_main": m.coef().get("pm25_7d", np.nan),
                "stress_main": m.coef().get(var, np.nan),
                "tmean_x_pm25_coef": m.coef().get("tmean_c:pm25_7d", np.nan),
                "tmean_x_pm25_p":    m.pvalue().get("tmean_c:pm25_7d", np.nan),
                "tmean_x_stress_coef": m.coef().get(f"tmean_c:{var}", np.nan),
                "tmean_x_stress_p":    m.pvalue().get(f"tmean_c:{var}", np.nan),
                "pm25_x_stress_coef":  m.coef().get(f"pm25_7d:{var}", np.nan),
                "pm25_x_stress_p":     m.pvalue().get(f"pm25_7d:{var}", np.nan),
                "three_way_coef": m.coef().get(three_way, np.nan),
                "three_way_p":    m.pvalue().get(three_way, np.nan),
                "n": int(m._N),
            })
        except Exception as e:
            print(f"  failed {var}: {e}")
            rows.append({"stressor": label, "var": var, "three_way_coef": np.nan, "three_way_p": np.nan, "n": 0})
    out = pd.DataFrame(rows).sort_values("three_way_p")
    print("\n=== TABLE G: heat × PM2.5(7d) × stressor (3-way) ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(RES / "tableG_three_way_pm25.csv", index=False)
    return out


def main() -> None:
    df = load_data()
    print(f"sample with PM2.5: {len(df):,}")

    table_e_pm25(df)
    table_f_event_study(df)
    table_g_three_way_pm25(df)


if __name__ == "__main__":
    main()
