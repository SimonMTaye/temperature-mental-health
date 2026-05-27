"""Build per-(pidlink, wave) stressor / covariate variables for IFLS4 and IFLS5.

Outputs: data/generated/stressors.parquet
  pidlink, wave, age, sex, married, widowed, edu_yrs, urban,
  hh_size, pce_log, pce_quintile,
  loan_rejected, agri_occupation,
  disaster_5yr, disaster_severe_5yr,
  hh_with_adl_limit (proxy: caregiver burden)

Notes
-----
- "PCE" for IFLS4 = pre-computed nominal monthly per-capita expenditure (pce07nom.dta).
- "PCE" for IFLS5 = computed here from b1_ks0/ks2/ks3 (food + non-food monthly).
- Disaster variables are HH-level (b2_nd1) — broadcast to all adults in HH.
- "agri_occupation" follows IFLS occupation codes 6-7 (agriculture/forestry/fishery)
  in primary job (b3a_tk2 tk24a / tk24a3); fallback to ar15a "did HHM work".
- Quintiles computed within wave so they're comparable across waves.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RAW = Path("E:/IFLS/extracted")
OUT = PROJECT / "data" / "generated"

# Make _sentinels.py importable when this script is run directly
sys.path.insert(0, str(Path(__file__).parent))
from _sentinels import clean_age, clean_categorical  # noqa: E402


def _ifls5_demographics() -> pd.DataFrame:
    ar = pd.read_stata(RAW / "IFLS5/hh14/bk_ar1.dta", convert_categoricals=False)
    ar = ar[["pidlink", "hhid14", "ar07", "ar09", "ar13", "ar16", "ar17"]].rename(columns={
        "ar07": "sex_raw", "ar09": "age", "ar13": "marital_raw",
        "ar16": "edu_lvl", "ar17": "edu_grade", "hhid14": "hhid",
    })
    ar = ar.dropna(subset=["pidlink"]).drop_duplicates("pidlink")
    ar["age"] = clean_age(ar.age)
    ar["marital_raw"] = clean_categorical(ar.marital_raw, digits=1)
    ar["sex"] = (ar.sex_raw == 3).map({True: "F", False: "M"})  # 1=M,3=F per IFLS docs
    ar["married"] = (ar.marital_raw == 2).astype(int)
    ar["widowed"] = (ar.marital_raw.isin([4, 5])).astype(int)  # 4=separated,5=widowed
    # crude years of education from level + grade
    lvl_map = {1: 0, 2: 6, 3: 9, 4: 9, 5: 12, 6: 12, 60: 12, 61: 12, 62: 12,
               63: 12, 11: 0, 14: 12, 15: 14, 17: 16, 13: 0, 95: 0, 98: 0}
    ar["edu_yrs"] = ar.edu_lvl.map(lvl_map).fillna(0).astype(int)
    return ar[["pidlink", "hhid", "age", "sex", "married", "widowed", "edu_yrs"]]


def _ifls4_demographics() -> pd.DataFrame:
    ar = pd.read_stata(RAW / "IFLS4/hh07/bk_ar1.dta", convert_categoricals=False)
    cols_keep = [c for c in ["pidlink", "hhid07", "ar07", "ar09", "ar13", "ar16", "ar17"] if c in ar.columns]
    ar = ar[cols_keep].rename(columns={
        "ar07": "sex_raw", "ar09": "age", "ar13": "marital_raw",
        "ar16": "edu_lvl", "ar17": "edu_grade", "hhid07": "hhid",
    })
    ar = ar.dropna(subset=["pidlink"]).drop_duplicates("pidlink")
    ar["age"] = clean_age(ar.age)
    if "marital_raw" in ar.columns:
        ar["marital_raw"] = clean_categorical(ar.marital_raw, digits=1)
    ar["sex"] = (ar.sex_raw == 3).map({True: "F", False: "M"})
    ar["married"] = (ar.get("marital_raw").eq(2)).astype(int) if "marital_raw" in ar.columns else 0
    ar["widowed"] = (ar.get("marital_raw").isin([4, 5])).astype(int) if "marital_raw" in ar.columns else 0
    lvl_map = {1: 0, 2: 6, 3: 9, 4: 9, 5: 12, 6: 12, 60: 12, 61: 12, 62: 12,
               63: 12, 11: 0, 14: 12, 15: 14, 17: 16, 13: 0, 95: 0, 98: 0}
    ar["edu_yrs"] = ar.edu_lvl.map(lvl_map).fillna(0).astype(int) if "edu_lvl" in ar.columns else 0
    return ar[["pidlink", "hhid", "age", "sex", "married", "widowed", "edu_yrs"]]


def _ifls5_pce() -> pd.DataFrame:
    """Approx monthly per-capita expenditure for IFLS5.
       weekly -> *4.33; monthly stays as is. Sums major modules.
       Quintile within IFLS5 sample.
    """
    # b1_ks0 has weekly food (ks02a) + monthly non-food various items at HH level
    ks0 = pd.read_stata(RAW / "IFLS5/hh14/b1_ks0.dta", convert_categoricals=False)
    # Aggregate any monetary monthly columns we can identify
    money_cols = [c for c in ks0.columns if c.startswith("ks") and ks0[c].dtype.kind in "if"]
    # ks02a/ks04b are weekly food; rest monthly
    weekly_food = ks0.get("ks02a", 0).fillna(0) * 4.33
    monthly_other = pd.Series(0.0, index=ks0.index)
    for c in money_cols:
        if c in ("ks02a", "ks04b"):
            continue
        monthly_other = monthly_other + ks0[c].fillna(0)
    ks0["x_total_mo"] = weekly_food + monthly_other

    # household size: count of pidlink in bk_ar1
    ar = pd.read_stata(RAW / "IFLS5/hh14/bk_ar1.dta", convert_categoricals=False)
    hhsize = ar.groupby("hhid14").size().rename("hhsize").reset_index()
    out = ks0[["hhid14", "x_total_mo"]].merge(hhsize, on="hhid14")
    out["pce"] = out.x_total_mo / out.hhsize.replace(0, np.nan)
    out["wave"] = "IFLS5"
    return out[["hhid14", "hhsize", "pce", "wave"]].rename(columns={"hhid14": "hhid"})


def _ifls4_pce() -> pd.DataFrame:
    df = pd.read_stata(RAW / "consumption/pce/pce07nom.dta", convert_categoricals=False)
    df = df[["hhid07", "hhsize", "pce"]].rename(columns={"hhid07": "hhid"})
    df["wave"] = "IFLS4"
    return df


def _disaster(wave: str) -> pd.DataFrame:
    folder = RAW / ("IFLS5/hh14" if wave == "IFLS5" else "IFLS4/hh07")
    nd = pd.read_stata(folder / "b2_nd1.dta", convert_categoricals=False)
    hhcol = "hhid14" if wave == "IFLS5" else "hhid07"
    nd["disaster_5yr"] = (nd.nd01 == 1).astype(int)
    nd["disaster_severe_5yr"] = (nd.nd02 == 1).astype(int)
    nd = nd[[hhcol, "disaster_5yr", "disaster_severe_5yr"]].rename(columns={hhcol: "hhid"})
    nd = nd.drop_duplicates("hhid")
    nd["wave"] = wave
    return nd


def _loan_rejected(wave: str) -> pd.DataFrame:
    folder = RAW / ("IFLS5/hh14" if wave == "IFLS5" else "IFLS4/hh07")
    bh = pd.read_stata(folder / "b2_bh.dta", convert_categoricals=False)
    hhcol = "hhid14" if wave == "IFLS5" else "hhid07"
    bh["loan_rejected"] = (bh.bh04 == 1).astype(int)
    bh = bh[[hhcol, "loan_rejected"]].rename(columns={hhcol: "hhid"}).drop_duplicates("hhid")
    bh["wave"] = wave
    return bh


def _agri_occupation(wave: str) -> pd.DataFrame:
    """Identify adults whose primary job is agriculture/forestry/fishery (sector 1)."""
    folder = RAW / ("IFLS5/hh14" if wave == "IFLS5" else "IFLS4/hh07")
    # tk2 has sector codes; tk24 series is the industry of primary job
    p = folder / "b3a_tk2.dta"
    if not p.exists():
        return pd.DataFrame(columns=["pidlink", "wave", "agri_occupation"])
    tk = pd.read_stata(p, convert_categoricals=False)
    sector_cols = [c for c in tk.columns if c.lower().startswith("tk19a") or c.lower().startswith("tk24")]
    # Heuristic: look for column whose values cluster around 1-9 (1-digit sector)
    # IFLS sector codes: 1 = agriculture/forestry/fishing
    candidate = None
    for c in sector_cols:
        u = tk[c].dropna().astype("Int64").unique()
        if len(u) and 1 in u and tk[c].between(1, 99).any():
            candidate = c
            break
    if candidate is None:
        return pd.DataFrame(columns=["pidlink", "wave", "agri_occupation"])
    tk["agri_occupation"] = (tk[candidate] == 1).astype(int)
    out = tk[["pidlink", "agri_occupation"]].drop_duplicates("pidlink")
    out["wave"] = wave
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Demographics
    demo = pd.concat([
        _ifls4_demographics().assign(wave="IFLS4"),
        _ifls5_demographics().assign(wave="IFLS5"),
    ], ignore_index=True)
    print(f"demographics: {len(demo):,}")

    # PCE
    pce = pd.concat([_ifls4_pce(), _ifls5_pce()], ignore_index=True)
    pce["pce"] = pce.pce.replace([0, np.inf, -np.inf], np.nan)
    pce["pce_log"] = np.log(pce.pce)
    pce["pce_quintile"] = pce.groupby("wave")["pce"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1
    )
    print(f"pce: {len(pce):,}")

    # HH-level shocks
    disaster = pd.concat([_disaster("IFLS4"), _disaster("IFLS5")], ignore_index=True)
    loan = pd.concat([_loan_rejected("IFLS4"), _loan_rejected("IFLS5")], ignore_index=True)
    print(f"disaster: {len(disaster):,}, loan: {len(loan):,}")

    # Individual-level
    agri = pd.concat([_agri_occupation("IFLS4"), _agri_occupation("IFLS5")], ignore_index=True)
    print(f"agri occupation: {len(agri):,}")

    # Merge by (pidlink, wave) — most are HH-level so first attach to demo via hhid
    out = demo.copy()
    out = out.merge(pce[["hhid", "wave", "hhsize", "pce", "pce_log", "pce_quintile"]],
                    on=["hhid", "wave"], how="left")
    out = out.merge(disaster, on=["hhid", "wave"], how="left")
    out = out.merge(loan, on=["hhid", "wave"], how="left")
    out = out.merge(agri, on=["pidlink", "wave"], how="left")

    for c in ["disaster_5yr", "disaster_severe_5yr", "loan_rejected", "agri_occupation"]:
        out[c] = out[c].fillna(0).astype(int)

    # Macro shock flags will be computed in the analysis script using interview_date
    out.to_parquet(OUT / "stressors.parquet", index=False)
    print(f"\nwrote {len(out):,} stressor rows to {OUT/'stressors.parquet'}")
    print("\nstressor prevalence by wave:")
    print(out.groupby("wave").agg(
        n=("pidlink", "size"),
        age_med=("age", "median"),
        female_pct=("sex", lambda s: 100 * (s == "F").mean()),
        edu_yrs_med=("edu_yrs", "median"),
        widowed_pct=("widowed", lambda s: 100 * s.mean()),
        pce_log_med=("pce_log", "median"),
        loan_rejected_pct=("loan_rejected", lambda s: 100 * s.mean()),
        disaster_5yr_pct=("disaster_5yr", lambda s: 100 * s.mean()),
        disaster_severe_pct=("disaster_severe_5yr", lambda s: 100 * s.mean()),
        agri_occ_pct=("agri_occupation", lambda s: 100 * s.mean()),
    ).round(2))


if __name__ == "__main__":
    main()
