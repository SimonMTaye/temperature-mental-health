"""Extract individual-level IFLS4 + IFLS5 records with interview date + admin codes.

For each wave, returns one row per (pidlink, wave) with:
  pidlink, hhid, wave, interview_date, prov_code, kab_code, kec_code

prov_code  = BPS 2-digit
kab_code   = BPS 4-digit  (prov*100 + within-province kab)
kec_code   = BPS 7-digit  (kab*1000 + within-kab kec)

Output: data/generated/individuals.parquet
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RAW = Path("E:/IFLS/extracted")
OUT = PROJECT / "data" / "generated"


def _ifls5_individuals() -> pd.DataFrame:
    # Interview date sits in b3a_time (one row per book-3A interview attempt)
    t = pd.read_stata(RAW / "IFLS5/hh14/b3a_time.dta", convert_categoricals=False)
    t = t[(t.ivwyr > 0) & (t.ivwmth > 0) & (t.ivwday > 0)].copy()
    t["interview_date"] = pd.to_datetime(
        dict(year=t.ivwyr.astype(int), month=t.ivwmth.astype(int), day=t.ivwday.astype(int)),
        errors="coerce",
    )
    t = t.dropna(subset=["interview_date"])
    # Take the FIRST successful visit per pidlink
    t = t.sort_values(["pidlink", "interview_date"]).drop_duplicates("pidlink", keep="first")
    t = t[["pidlink", "hhid14", "interview_date"]].rename(columns={"hhid14": "hhid"})

    # Admin codes from bk_sc1 (HH-level)
    sc = pd.read_stata(RAW / "IFLS5/hh14/bk_sc1.dta", convert_categoricals=False)
    sc = sc[["hhid14", "sc01_14_14", "sc02_14_14", "sc03_14_14"]].rename(
        columns={"hhid14": "hhid", "sc01_14_14": "prov", "sc02_14_14": "kab_in_prov", "sc03_14_14": "kec_in_kab"}
    )
    sc = sc.dropna(subset=["prov", "kab_in_prov", "kec_in_kab"]).drop_duplicates("hhid")
    sc["prov"] = sc.prov.astype(int)
    sc["kab_in_prov"] = sc.kab_in_prov.astype(int)
    sc["kec_in_kab"] = sc.kec_in_kab.astype(int)
    sc["prov_code"] = sc.prov
    sc["kab_code"] = sc.prov * 100 + sc.kab_in_prov
    sc["kec_code"] = sc.kab_code * 1000 + sc.kec_in_kab

    out = t.merge(sc[["hhid", "prov_code", "kab_code", "kec_code"]], on="hhid", how="inner")
    out["wave"] = "IFLS5"
    return out


def _ifls4_individuals() -> pd.DataFrame:
    # b3a_cov has ivwday1/ivwmth1/ivwyr1 (year is 2-digit: 7=2007, 8=2008)
    cov = pd.read_stata(RAW / "IFLS4/hh07/b3a_cov.dta", convert_categoricals=False)
    cov = cov[(cov.ivwyr1 > 0) & (cov.ivwmth1 > 0) & (cov.ivwday1 > 0)].copy()
    cov["yr"] = cov.ivwyr1.astype(int) + 2000
    cov["interview_date"] = pd.to_datetime(
        dict(year=cov.yr, month=cov.ivwmth1.astype(int), day=cov.ivwday1.astype(int)),
        errors="coerce",
    )
    cov = cov.dropna(subset=["interview_date"])
    cov = cov.sort_values(["pidlink", "interview_date"]).drop_duplicates("pidlink", keep="first")
    cov = cov[["pidlink", "hhid07", "interview_date"]].rename(columns={"hhid07": "hhid"})

    # Admin codes from bk_sc — use 2007 codes (sc010707/sc020707/sc030707)
    sc = pd.read_stata(RAW / "IFLS4/hh07/bk_sc.dta", convert_categoricals=False)
    sc = sc[["hhid07", "sc010707", "sc020707", "sc030707"]].rename(
        columns={"hhid07": "hhid", "sc010707": "prov", "sc020707": "kab_in_prov", "sc030707": "kec_in_kab"}
    )
    sc = sc.dropna(subset=["prov", "kab_in_prov", "kec_in_kab"]).drop_duplicates("hhid")
    sc["prov"] = sc.prov.astype(int)
    sc["kab_in_prov"] = sc.kab_in_prov.astype(int)
    sc["kec_in_kab"] = sc.kec_in_kab.astype(int)
    sc["prov_code"] = sc.prov
    sc["kab_code"] = sc.prov * 100 + sc.kab_in_prov
    sc["kec_code"] = sc.kab_code * 1000 + sc.kec_in_kab

    out = cov.merge(sc[["hhid", "prov_code", "kab_code", "kec_code"]], on="hhid", how="inner")
    out["wave"] = "IFLS4"
    return out


def main() -> None:
    parts = [_ifls4_individuals(), _ifls5_individuals()]
    out = pd.concat(parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "individuals.parquet", index=False)
    print(f"wrote {len(out):,} individual-wave rows to {OUT/'individuals.parquet'}")
    print(out.groupby("wave").agg(
        n=("pidlink", "size"),
        date_min=("interview_date", "min"),
        date_max=("interview_date", "max"),
        n_provinces=("prov_code", "nunique"),
        n_kab=("kab_code", "nunique"),
        n_kec=("kec_code", "nunique"),
    ))


if __name__ == "__main__":
    main()
