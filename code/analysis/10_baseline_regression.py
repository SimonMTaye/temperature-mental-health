"""Baseline regressions: does daily temperature on the interview day correlate with CES-D?
Then heterogeneity by every feasible stressor.

Specifications all use:
  - DV: cesd_raw (continuous 0-30) and depressed (binary, CES-D>=10) as robustness
  - FE: kab (kabupaten) + month-of-year + year + wave
  - SE: clustered at kabupaten level
  - Sample: pooled IFLS4 + IFLS5 adults

Heterogeneity: for each stressor S, run cesd_raw ~ tmean + S + tmean*S + controls + FE.

Key questions answered
---------------------
1. Does daily mean temp move CES-D? (sign and magnitude)
2. Is the effect bigger for hotter values? (check tmax, heat index, anomaly)
3. Which stressed subgroups react more? (rank β_interaction)
4. Lead-of-temperature placebo - should be near zero
5. Pre-vs-post Jokowi subsidy cut & 2015 haze: are stressed periods more reactive?
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
RES = PROJECT / "data" / "generated" / "results"
RES.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(OUT / "analysis_dataset.parquet")
    # Drop rows with missing key covariates so all specs run on the same sample
    keep = ["cesd_raw", "tmean_c", "kabupaten_code", "wave", "month", "year",
            "age", "sex", "edu_yrs", "married", "widowed",
            "disaster_5yr", "disaster_severe_5yr", "loan_rejected", "agri_occupation"]
    df = df.dropna(subset=keep)
    df["female"] = (df.sex == "F").astype(int)
    df["wave_num"] = (df.wave == "IFLS5").astype(int)
    return df


def baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1: progression of FE / controls."""
    fits = []
    specs = [
        ("M1: pooled, no FE",
         "cesd_raw ~ tmean_c"),
        ("M2: + wave + month + year FE",
         "cesd_raw ~ tmean_c | wave + month + year"),
        ("M3: + kab FE",
         "cesd_raw ~ tmean_c | wave + month + year + kabupaten_code"),
        ("M4: + demographic controls",
         "cesd_raw ~ tmean_c + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("M5: tmax instead of tmean",
         "cesd_raw ~ tmax_c + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("M6: tmin (nighttime) instead",
         "cesd_raw ~ tmin_c + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("M7: heat index",
         "cesd_raw ~ heat_idx_c + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("M8: 30-day temperature anomaly",
         "cesd_raw ~ t_anom_today + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("M9: lead7 placebo",
         "cesd_raw ~ tmean_lead7 + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
    ]
    rows = []
    for name, formula in specs:
        try:
            m = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_code"})
            heat_term = formula.split("~")[1].split("+")[0].strip().split("|")[0].strip()
            row = {
                "spec": name,
                "term": heat_term,
                "coef": m.coef()[heat_term],
                "se": m.se()[heat_term],
                "tstat": m.tstat()[heat_term],
                "pval": m.pvalue()[heat_term],
                "n": int(m._N),
            }
            rows.append(row)
        except Exception as e:
            rows.append({"spec": name, "term": "ERROR", "coef": np.nan, "se": np.nan,
                          "tstat": np.nan, "pval": np.nan, "n": 0})
            print(f"  failed {name}: {e}")
    out = pd.DataFrame(rows)
    print("\n=== TABLE 1: baseline (DV = cesd_raw 0-30) ===")
    print(out.to_string(index=False))
    out.to_csv(RES / "table1_baseline.csv", index=False)
    return out


def heterogeneity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table 2: tmean × stressor interactions."""
    stressors = [
        ("disaster_severe_5yr",  "Severe disaster in last 5y"),
        ("disaster_5yr",         "Any disaster in last 5y"),
        ("loan_rejected",        "Loan turned down (12mo)"),
        ("agri_occupation",      "Agriculture/forestry primary job"),
        ("widowed",              "Widowed/separated"),
        ("haze_2015",            "Sept-Nov 2015 haze months"),
        ("post_subsidy",         "Post Nov-2014 fuel-subsidy cut"),
    ]
    # PCE quintile bottom = treat as high-stress
    df = df.copy()
    df["pce_q1"] = (df.pce_quintile == 1).astype(int)
    df["age_60p"] = (df.age >= 60).astype(int)
    df["female_int"] = df.female  # already 0/1
    stressors += [
        ("pce_q1",      "Bottom expenditure quintile"),
        ("age_60p",     "Age 60+"),
        ("female_int",  "Female"),
    ]

    rows = []
    base_ctrl = "+ age + edu_yrs + married + widowed"
    for var, label in stressors:
        formula = f"cesd_raw ~ tmean_c * {var} + female {base_ctrl} | wave + month + year + kabupaten_code"
        try:
            m = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_code"})
            inter = f"tmean_c:{var}"
            rows.append({
                "stressor": label,
                "var": var,
                "main_temp_coef": m.coef()["tmean_c"],
                "main_temp_se": m.se()["tmean_c"],
                "stress_main_coef": m.coef()[var],
                "stress_main_se": m.se()[var],
                "interaction_coef": m.coef()[inter],
                "interaction_se": m.se()[inter],
                "interaction_t": m.tstat()[inter],
                "interaction_p": m.pvalue()[inter],
                "n": int(m._N),
                "stress_share": float(df[var].mean()),
            })
        except Exception as e:
            print(f"  failed {var}: {e}")
            rows.append({"stressor": label, "var": var, "interaction_coef": np.nan,
                          "interaction_p": np.nan, "n": 0})
    out = pd.DataFrame(rows)
    out = out.sort_values("interaction_p")
    print("\n=== TABLE 2: heterogeneity (DV = cesd_raw, β on tmean_c × stressor) ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(RES / "table2_heterogeneity.csv", index=False)
    return out


def binary_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Same baseline but DV = depressed (CES-D >= 10)."""
    rows = []
    for label, formula in [
        ("Linear prob: depressed ~ tmean",
         "depressed ~ tmean_c + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("Linear prob: depressed ~ tmax",
         "depressed ~ tmax_c + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
        ("Linear prob: depressed ~ heat_idx",
         "depressed ~ heat_idx_c + age + female + edu_yrs + married + widowed | wave + month + year + kabupaten_code"),
    ]:
        m = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_code"})
        term = formula.split("~")[1].split("+")[0].strip()
        rows.append({"spec": label, "coef": m.coef()[term], "se": m.se()[term],
                      "p": m.pvalue()[term], "n": int(m._N)})
    out = pd.DataFrame(rows)
    print("\n=== TABLE 3: robustness DV=depressed (≥10) ===")
    print(out.round(5).to_string(index=False))
    out.to_csv(RES / "table3_binary.csv", index=False)
    return out


def main() -> None:
    df = load_data()
    print(f"analysis sample: {len(df):,}")
    print(f"  CES-D raw: mean={df.cesd_raw.mean():.2f}  sd={df.cesd_raw.std():.2f}")
    print(f"  tmean: mean={df.tmean_c.mean():.2f}°C  sd={df.tmean_c.std():.2f}°C")

    baseline_table(df)
    heterogeneity_table(df)
    binary_robustness(df)


if __name__ == "__main__":
    main()
