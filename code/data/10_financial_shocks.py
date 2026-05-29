"""Build financial-hardship shock variables for IFLS4 and IFLS5.

Adds these per (pidlink, wave):
  recent_job_loss_5y     1 if respondent had ≥1 job termination in past 5 years (b3a_tk4.tk46c)
  involuntary_loss_5y    1 if last termination reason was lay-off/closure (tk46m in {1,2,3,7,9})
  days_since_last_loss   days between interview_date and last termination
  job_loss_within_yr     1 if last termination within 365 days of interview

  vehicle_owner          HH owns vehicle (b2_hr1, hrtype='E', hr01==1)
  urban                  HH in urban area (bk_sc1.sc05==1 in IFLS5; equivalent in IFLS4)

  cash_transfer_recipient  HH receives PKH/BLT/cash transfer (b1_ksr1.ksr17==1 OR b2_kr.kr27b==1)
  health_card              HH has Jamkesmas/BPJS/health card (b2_kr.kr26==1)

  palm_region              province_code in (Sumatra+Kalimantan palm-oil heartland)
  palm_price_usd_mt        World Bank monthly palm oil price for interview month
  palm_price_z             z-scored palm price (within 2007-2016 history)

Output: data/generated/financial_shocks.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import OUT, RAW  # noqa: E402
from _sentinels import clean_count, clean_month, clean_year  # noqa: E402
from _ifls_wave import hhid_col, wave_folder  # noqa: E402
from _schemas import FINANCIAL_SHOCKS_SCHEMA  # noqa: E402

# Provinces with significant palm oil cultivation (Indonesian Palm Oil Producers Association)
PALM_PROVS = {
    11, 12, 13, 14, 15, 16, 17, 18, 19, 21,  # Sumatra
    61, 62, 63, 64,                          # Kalimantan
}

# World Bank Pink Sheet monthly palm oil price (USD/MT, "Palm oil, Malaysia and Indonesia origin")
# Covers IFLS4 + IFLS5 fielding windows. Values approximate published series.
PALM_PRICE = {
    (2007, 6): 850, (2007, 7): 879, (2007, 8): 829, (2007, 9): 866, (2007,10): 861,
    (2007,11): 950, (2007,12): 1030,
    (2008, 1):1075, (2008, 2):1188, (2008, 3):1306, (2008, 4):1180, (2008, 5):1234,
    (2008, 6):1199, (2008, 7):1119, (2008, 8): 856, (2008, 9): 706,
    (2014, 8): 745, (2014, 9): 695, (2014,10): 696, (2014,11): 712, (2014,12): 715,
    (2015, 1): 678, (2015, 2): 651, (2015, 3): 657, (2015, 4): 660, (2015, 5): 656,
    (2015, 6): 658, (2015, 7): 627, (2015, 8): 528, (2015, 9): 511, (2015,10): 528,
    (2015,11): 529, (2015,12): 549,
}


def _job_loss_from_df(df: pd.DataFrame, *, wave: str) -> pd.DataFrame:
    out = pd.DataFrame({"pidlink": df.pidlink})
    # Clean sentinels in the job-loss recall fields before deriving anything
    tk46c_clean = clean_count(df.tk46c, max_real=50)        # masks 98/99/998/999
    out["recent_job_loss_5y"] = (tk46c_clean.fillna(0) >= 1).astype(int)
    # tk46m reason codes: 1=fired, 2=laid off, 3=plant closed, 7=health, 9=other
    # Treat 1,2,3 as INVOLUNTARY (forced loss)
    out["involuntary_loss_5y"] = df.tk46m.isin([1, 2, 3]).astype(int)
    # Date of last termination (we'll compute days_since later when we have interview_date).
    # Mask sentinels so a "don't know" date doesn't get clipped into a fake December.
    out["last_loss_year"] = clean_year(df.tk46dy)
    out["last_loss_month"] = clean_month(df.tk46dm)
    out["wave"] = wave
    return out.drop_duplicates("pidlink")


def _job_loss(wave: str) -> pd.DataFrame:
    df = pd.read_stata(wave_folder(RAW, wave) / "b3a_tk4.dta", convert_categoricals=False)
    return _job_loss_from_df(df, wave=wave)


def _vehicle_owner_from_df(hr: pd.DataFrame, *, hhid_col_name: str, wave: str) -> pd.DataFrame:
    veh = hr[hr.hrtype == "E"].copy()
    veh["vehicle_owner"] = (veh.hr01 == 1).astype(int)
    out = veh[[hhid_col_name, "vehicle_owner"]].rename(columns={hhid_col_name: "hhid"}).drop_duplicates("hhid")
    out["wave"] = wave
    return out


def _vehicle_owner(wave: str) -> pd.DataFrame:
    hr = pd.read_stata(wave_folder(RAW, wave) / "b2_hr1.dta", convert_categoricals=False)
    return _vehicle_owner_from_df(hr, hhid_col_name=hhid_col(wave), wave=wave)


def _urban_from_df(screening: pd.DataFrame, *, hhid_col_name: str, wave: str) -> pd.DataFrame:
    screening["urban"] = (screening.sc05 == 1).astype(int)  # 1=Urban, 2=Rural in IFLS convention
    out = screening[[hhid_col_name, "urban"]].rename(columns={hhid_col_name: "hhid"}).drop_duplicates("hhid")
    out["wave"] = wave
    return out


def _urban(wave: str) -> pd.DataFrame:
    fname = "bk_sc1.dta" if wave == "IFLS5" else "bk_sc.dta"
    screening = pd.read_stata(wave_folder(RAW, wave) / fname, convert_categoricals=False)
    return _urban_from_df(screening, hhid_col_name=hhid_col(wave), wave=wave)


def _cash_transfer_from_frames(
    *,
    wave: str,
    hhid_col_name: str,
    ksr: pd.DataFrame | None,
    kr: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []
    # Source 1: ksr17 (any HH cash transfer received)
    if ksr is not None and "ksr17" in ksr.columns:
        rec = ksr.groupby(hhid_col_name)["ksr17"].apply(lambda x: int((x == 1).any())).reset_index()
        rec.columns = ["hhid", "any_cash_transfer_ksr"]
        rows.append(rec)
    # Source 2: kr27b BLT card (IFLS5 only)
    if kr is not None:
        cols = {"hhid": kr[hhid_col_name]}
        if "kr27b" in kr.columns:
            cols["blt_card"] = (kr.kr27b == 1).astype(int)
        if "kr26" in kr.columns:
            cols["health_card"] = (kr.kr26 == 1).astype(int)
        rows.append(pd.DataFrame(cols).drop_duplicates("hhid"))
    if not rows:
        return pd.DataFrame(columns=["hhid", "wave"])
    out = rows[0]
    for r in rows[1:]:
        out = out.merge(r, on="hhid", how="outer")
    for c in ["any_cash_transfer_ksr", "blt_card", "health_card"]:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)
    out["cash_transfer_recipient"] = out.filter(items=[
        "any_cash_transfer_ksr", "blt_card",
    ]).max(axis=1).fillna(0).astype(int)
    out["wave"] = wave
    return out


def _cash_transfer(wave: str) -> pd.DataFrame:
    folder = wave_folder(RAW, wave)
    ksr_path = folder / "b1_ksr1.dta"
    kr_path = folder / "b2_kr.dta"
    return _cash_transfer_from_frames(
        wave=wave,
        hhid_col_name=hhid_col(wave),
        ksr=pd.read_stata(ksr_path, convert_categoricals=False) if ksr_path.exists() else None,
        kr=pd.read_stata(kr_path, convert_categoricals=False) if kr_path.exists() else None,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Per-individual: job loss
    jl = pd.concat([_job_loss("IFLS4"), _job_loss("IFLS5")], ignore_index=True)
    print(f"job loss rows: {len(jl):,}; recent_job_loss_5y mean={jl.recent_job_loss_5y.mean():.3f}")

    # Per-HH: vehicle ownership, urban, cash transfer
    veh = pd.concat([_vehicle_owner("IFLS4"), _vehicle_owner("IFLS5")], ignore_index=True)
    urb = pd.concat([_urban("IFLS4"), _urban("IFLS5")], ignore_index=True)
    cash = pd.concat([_cash_transfer("IFLS4"), _cash_transfer("IFLS5")], ignore_index=True)
    print(f"vehicle: {len(veh):,};  urban: {len(urb):,};  cash: {len(cash):,}")

    # Merge to individual via stressors.parquet (which has pidlink↔hhid mapping)
    stressors = pd.read_parquet(OUT / "stressors.parquet")
    base = stressors[["pidlink", "wave", "hhid"]].drop_duplicates(["pidlink", "wave"])
    out = base.merge(jl, on=["pidlink", "wave"], how="left")
    out = out.merge(veh, on=["hhid", "wave"], how="left")
    out = out.merge(urb, on=["hhid", "wave"], how="left")
    out = out.merge(cash, on=["hhid", "wave"], how="left")

    # Add days_since_last_loss (need interview date)
    ind = pd.read_parquet(OUT / "individuals.parquet")[["pidlink", "wave", "interview_date", "province_code"]]
    out = out.merge(ind, on=["pidlink", "wave"], how="left")
    has_date = out.last_loss_year.notna() & (out.last_loss_year > 0) & out.last_loss_month.notna() & (out.last_loss_month > 0)
    out["last_loss_date"] = pd.NaT
    if has_date.sum() > 0:
        # tk46dy in IFLS4 might be 2-digit; pad
        yr = out.loc[has_date, "last_loss_year"].astype(int)
        yr = np.where(yr < 100, yr + 2000, yr)
        m = out.loc[has_date, "last_loss_month"].astype(int).clip(1, 12)
        out.loc[has_date, "last_loss_date"] = pd.to_datetime(
            dict(year=yr, month=m, day=15), errors="coerce"
        )
    out["last_loss_date"] = pd.to_datetime(out["last_loss_date"])
    out["days_since_last_loss"] = (out.interview_date - out.last_loss_date).dt.days
    out["job_loss_within_yr"] = ((out.days_since_last_loss >= 0) & (out.days_since_last_loss <= 365)).astype(int)

    # Palm region + monthly price
    out["palm_region"] = out.province_code.isin(PALM_PROVS).astype(int)
    out["intvw_yr"] = out.interview_date.dt.year
    out["intvw_mo"] = out.interview_date.dt.month
    out["palm_price_usd_mt"] = out.apply(
        lambda r: PALM_PRICE.get((r.intvw_yr, r.intvw_mo), np.nan), axis=1
    )
    # z-score within union of months in our table
    pp = pd.Series(PALM_PRICE.values())
    out["palm_price_z"] = (out.palm_price_usd_mt - pp.mean()) / pp.std()
    # palm shock: low price (z<0) × in palm region
    out["palm_shock"] = (out.palm_region * (-out.palm_price_z.fillna(0))).clip(lower=0)

    keep = ["pidlink", "wave",
            "recent_job_loss_5y", "involuntary_loss_5y", "days_since_last_loss",
            "job_loss_within_yr",
            "vehicle_owner", "urban",
            "cash_transfer_recipient", "blt_card", "health_card",
            "palm_region", "palm_price_usd_mt", "palm_price_z", "palm_shock"]
    keep = [c for c in keep if c in out.columns]
    out_final = out[keep].copy()
    for c in ["blt_card", "health_card"]:
        if c not in out_final.columns:
            out_final[c] = 0
    out_final = out_final[
        ["pidlink", "wave",
         "recent_job_loss_5y", "involuntary_loss_5y", "days_since_last_loss",
         "job_loss_within_yr",
         "vehicle_owner", "urban",
         "cash_transfer_recipient", "blt_card", "health_card",
         "palm_region", "palm_price_usd_mt", "palm_price_z", "palm_shock"]
    ]
    # Cast numeric flags
    for c in ["recent_job_loss_5y", "involuntary_loss_5y", "job_loss_within_yr",
              "vehicle_owner", "urban", "cash_transfer_recipient",
              "blt_card", "health_card", "palm_region"]:
        if c in out_final.columns:
            out_final[c] = out_final[c].fillna(0).astype(int)

    out_final = FINANCIAL_SHOCKS_SCHEMA.validate(out_final)
    out_final.to_parquet(OUT / "financial_shocks.parquet", index=False)
    print(f"\nwrote {len(out_final):,} rows to {OUT/'financial_shocks.parquet'}")
    print("\nshock prevalence by wave:")
    summ = out_final.groupby("wave").agg(
        n=("pidlink", "size"),
        recent_loss_pct=("recent_job_loss_5y", lambda s: 100*s.mean()),
        invol_loss_pct=("involuntary_loss_5y", lambda s: 100*s.mean()),
        job_loss_yr_pct=("job_loss_within_yr", lambda s: 100*s.mean()),
        vehicle_pct=("vehicle_owner", lambda s: 100*s.mean()),
        urban_pct=("urban", lambda s: 100*s.mean()),
        cash_xfer_pct=("cash_transfer_recipient", lambda s: 100*s.mean()),
        health_card_pct=("health_card", lambda s: 100*s.mean()) if "health_card" in out_final.columns else ("recent_job_loss_5y", "size"),
        palm_region_pct=("palm_region", lambda s: 100*s.mean()),
    ).round(2)
    print(summ)


if __name__ == "__main__":
    main()
