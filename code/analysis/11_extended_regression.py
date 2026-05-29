"""Extended regression suite. Tries:
  A. Non-linear heat features: hot-day counts (>30°C, >32°C) over 7d / 30d windows;
     CDD; tmax + heat-bin specs.
  B. AOD (haze) main effect + heat × AOD interaction.
  C. 2015-haze focused subsample (Sumatra + Kalimantan + Java).
  D. Heat × AOD × already-stressed (3-way interaction).
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
    aod = pd.read_parquet(OUT / "aod_monthly_kab.parquet")
    df["female"] = (df.sex == "F").astype(int)
    df["pce_q1"] = (df.pce_quintile == 1).astype(int)
    df["age_60p"] = (df.age >= 60).astype(int)

    # Attach AOD (kab × interview month-year)
    df["intvw_yr"] = df.interview_date.dt.year
    df["intvw_mo"] = df.interview_date.dt.month
    df = df.merge(
        aod.rename(columns={"year": "intvw_yr", "month": "intvw_mo"}),
        on=["kabupaten_code", "intvw_yr", "intvw_mo"], how="left",
    )

    # Re-attach the heat-features that 06_build_analysis didn't carry forward
    temp = pd.read_parquet(OUT / "daily_temperature_kab.parquet")
    temp["date"] = pd.to_datetime(temp.date)
    feat_cols = ["kabupaten_code", "date", "hot28", "hot30", "hot32",
                 "hot28_7d", "hot30_7d", "hot32_7d",
                 "hot28_30d", "hot30_30d", "hot32_30d",
                 "cdd", "cdd_7d", "is_extreme", "tmean_p90_30d"]
    df = df.merge(
        temp[feat_cols],
        left_on=["kabupaten_code", "interview_date"], right_on=["kabupaten_code", "date"],
        how="left",
    ).drop(columns=["date"])

    keep = ["cesd_raw", "tmean_c", "tmax_c", "kabupaten_code", "wave", "month", "year",
            "age", "sex", "edu_yrs", "married", "widowed",
            "disaster_severe_5yr", "loan_rejected", "agri_occupation",
            "hot30_7d", "hot32_7d", "hot30_30d", "cdd_7d", "is_extreme"]
    df = df.dropna(subset=keep)
    return df


def table_a_nonlinear(df: pd.DataFrame) -> pd.DataFrame:
    """Non-linear heat features."""
    rows = []
    specs = [
        ("Hot days (>30°C tmax) past 7d",     "hot30_7d"),
        ("Hot days (>32°C tmax) past 7d",     "hot32_7d"),
        ("Hot days (>30°C tmax) past 30d",    "hot30_30d"),
        ("Hot days (>32°C tmax) past 30d",    "hot32_30d"),
        ("Cooling-degree-days (base 22) 7d",  "cdd_7d"),
        ("Today is local extreme (>p90)",     "is_extreme"),
        ("Tmax today (continuous)",           "tmax_c"),
        ("Tmean today (baseline)",            "tmean_c"),
    ]
    for label, term in specs:
        formula = f"cesd_raw ~ {term} + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"
        try:
            m = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_code"})
            rows.append({
                "spec": label, "term": term,
                "coef": m.coef()[term], "se": m.se()[term],
                "tstat": m.tstat()[term], "pval": m.pvalue()[term],
                "n": int(m._N),
            })
        except Exception as e:
            rows.append({"spec": label, "term": term, "coef": np.nan, "se": np.nan,
                          "tstat": np.nan, "pval": np.nan, "n": 0})
            print(f"  failed {label}: {e}")
    out = pd.DataFrame(rows).sort_values("pval")
    print("\n=== TABLE A: non-linear heat features (DV = cesd_raw) ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(RES / "tableA_nonlinear_heat.csv", index=False)
    return out


def table_b_aod(df: pd.DataFrame) -> pd.DataFrame:
    """AOD main effect + heat × AOD."""
    print("\n--- AOD coverage ---")
    print(f"  with AOD: {df.aod.notna().sum()} / {len(df)}")
    print(f"  AOD distribution:")
    print(df.aod.describe().round(3))

    rows = []
    specs = [
        ("AOD main effect",
         "cesd_raw ~ aod + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("Tmean × AOD",
         "cesd_raw ~ tmean_c * aod + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("Tmax × AOD",
         "cesd_raw ~ tmax_c * aod + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("Hot30_7d × AOD",
         "cesd_raw ~ hot30_7d * aod + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
    ]
    df_a = df.dropna(subset=["aod"])
    for label, formula in specs:
        try:
            m = pf.feols(formula, data=df_a, vcov={"CRV1": "kabupaten_code"})
            terms = list(m.coef().index)
            interaction_terms = [t for t in terms if ":" in t]
            rec = {"spec": label, "n": int(m._N)}
            for t in terms:
                if t.startswith(("aod", "tmean_c", "tmax_c", "hot30_7d")) or ":" in t:
                    rec[f"{t}_coef"] = m.coef()[t]
                    rec[f"{t}_p"] = m.pvalue()[t]
            rows.append(rec)
        except Exception as e:
            print(f"  failed {label}: {e}")
            rows.append({"spec": label, "n": 0})
    out = pd.DataFrame(rows)
    print("\n=== TABLE B: AOD + heat × AOD (DV = cesd_raw) ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(RES / "tableB_aod.csv", index=False)
    return out


def table_c_haze_subsample(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to 2015 haze-affected regions (Sumatra + Kalimantan + Java) and IFLS5 only."""
    HAZE_PROVS = list(range(11, 22)) + list(range(31, 37)) + list(range(61, 65))
    sub = df[(df.wave == "IFLS5") & (df.province_code.isin(HAZE_PROVS))].copy()
    print(f"\n--- Haze subsample (IFLS5, Sumatra+Java+Kalimantan): n={len(sub)} ---")
    print(f"  AOD coverage: {sub.aod.notna().sum()} / {len(sub)}")

    rows = []
    specs = [
        ("Tmean × AOD (haze subsample)",
         "cesd_raw ~ tmean_c * aod + age + female + edu_yrs + married + widowed | month + year + kabupaten_code"),
        ("Hot30_7d × AOD (haze subsample)",
         "cesd_raw ~ hot30_7d * aod + age + female + edu_yrs + married + widowed | month + year + kabupaten_code"),
        ("Tmean × log(AOD)",
         "cesd_raw ~ tmean_c * I(np.log(aod)) + age + female + edu_yrs + married + widowed | month + year + kabupaten_code"),
        ("Tmean × AOD × bottom-PCE (3-way)",
         "cesd_raw ~ tmean_c * aod * pce_q1 + age + female + edu_yrs + married + widowed | month + year + kabupaten_code"),
    ]
    sub_a = sub.dropna(subset=["aod"])
    for label, formula in specs:
        try:
            m = pf.feols(formula, data=sub_a, vcov={"CRV1": "kabupaten_code"})
            rec = {"spec": label, "n": int(m._N)}
            for t in m.coef().index:
                if any(k in t for k in ["aod", "tmean", "hot30", "pce_q1"]):
                    rec[f"{t}_coef"] = m.coef()[t]
                    rec[f"{t}_p"] = m.pvalue()[t]
            rows.append(rec)
        except Exception as e:
            print(f"  failed {label}: {e}")
            rows.append({"spec": label, "n": 0})
    out = pd.DataFrame(rows)
    print("\n=== TABLE C: 2015 haze focused (DV = cesd_raw) ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(RES / "tableC_haze_subsample.csv", index=False)
    return out


def table_d_three_way(df: pd.DataFrame) -> pd.DataFrame:
    """Heat × AOD × stressor 3-way interactions. The 'tipping point' hypothesis."""
    rows = []
    df_a = df.dropna(subset=["aod"])
    stressors = [
        ("pce_q1", "Bottom expenditure quintile"),
        ("disaster_severe_5yr", "Severe disaster in 5y"),
        ("agri_occupation", "Agriculture occupation"),
        ("widowed", "Widowed/separated"),
        ("age_60p", "Age 60+"),
        ("loan_rejected", "Loan rejected"),
    ]
    for var, label in stressors:
        formula = f"cesd_raw ~ tmean_c * aod * {var} + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"
        try:
            m = pf.feols(formula, data=df_a, vcov={"CRV1": "kabupaten_code"})
            three_way = f"tmean_c:aod:{var}"
            rows.append({
                "stressor": label,
                "var": var,
                "tmean_main": m.coef().get("tmean_c", np.nan),
                "tmean_se":   m.se().get("tmean_c", np.nan),
                "aod_main":   m.coef().get("aod", np.nan),
                "stress_main": m.coef().get(var, np.nan),
                "tmean_x_aod_coef": m.coef().get("tmean_c:aod", np.nan),
                "tmean_x_aod_p":    m.pvalue().get("tmean_c:aod", np.nan),
                "tmean_x_stress_coef": m.coef().get(f"tmean_c:{var}", np.nan),
                "tmean_x_stress_p":    m.pvalue().get(f"tmean_c:{var}", np.nan),
                "aod_x_stress_coef":   m.coef().get(f"aod:{var}", np.nan),
                "aod_x_stress_p":      m.pvalue().get(f"aod:{var}", np.nan),
                "three_way_coef": m.coef().get(three_way, np.nan),
                "three_way_p":    m.pvalue().get(three_way, np.nan),
                "n": int(m._N),
            })
        except Exception as e:
            print(f"  failed {var}: {e}")
            rows.append({"stressor": label, "var": var, "three_way_coef": np.nan,
                          "three_way_p": np.nan, "n": 0})
    out = pd.DataFrame(rows).sort_values("three_way_p")
    print("\n=== TABLE D: heat × AOD × stressor (3-way) ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(RES / "tableD_three_way.csv", index=False)
    return out


def main() -> None:
    df = load_data()
    print(f"sample: {len(df):,}")

    table_a_nonlinear(df)
    table_b_aod(df)
    table_c_haze_subsample(df)
    table_d_three_way(df)


if __name__ == "__main__":
    main()
