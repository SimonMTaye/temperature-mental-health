"""Build additional financial-distress stressors not yet tested.

  high_debt        : top-quartile of HH debt within wave.
                     IFLS5 uses b2_bh.bh28  (current outstanding loan stock).
                     IFLS4 uses b2_bh.bh10  (loan amount taken out in past 12 mo — a flow).
                     Each wave's quartile is measured WITHIN that wave, so wave FE absorbs
                     the conceptual difference between stock and flow.
  high_medical_oop : top-quartile (within wave) of hospitalisation OOP cost (b3b_rn1.rn19),
                     among adults who were hospitalised in past 12 months. Zero otherwise.
  pce_decline_q4   : panel respondents (in both IFLS4 and IFLS5) whose per-capita expenditure
                     (real, after deflator) fell into the bottom quartile of inter-wave change.

Output: data/generated/finance_distress_shocks.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import OUT, RAW  # noqa: E402
from _sentinels import clean_money as _clean_money  # noqa: E402
from _ifls_wave import hhid_col, wave_folder  # noqa: E402
from _schemas import FINANCE_DISTRESS_SHOCKS_SCHEMA  # noqa: E402


def _debt_from_bh(bh: pd.DataFrame, *, hhid_col_name: str, debt_col: str, wave: str) -> pd.DataFrame:
    """Normalize one wave's debt proxy to HH-wave rows."""
    bh["debt"] = _clean_money(bh[debt_col])
    out = bh[[hhid_col_name, "debt"]].rename(columns={hhid_col_name: "hhid"})
    out["wave"] = wave
    return out


def _high_debt() -> pd.DataFrame:
    """Top-quartile (within wave) of HH debt."""
    out_rows = [
        # IFLS5: bh28 = total outstanding loan stock
        _debt_from_bh(
            pd.read_stata(wave_folder(RAW, "IFLS5") / "b2_bh.dta", convert_categoricals=False),
            hhid_col_name=hhid_col("IFLS5"),
            debt_col="bh28",
            wave="IFLS5",
        ),
        # IFLS4: bh10 = loan amount in past 12 months (flow). Closest available proxy.
        _debt_from_bh(
            pd.read_stata(wave_folder(RAW, "IFLS4") / "b2_bh.dta", convert_categoricals=False),
            hhid_col_name=hhid_col("IFLS4"),
            debt_col="bh10",
            wave="IFLS4",
        ),
    ]
    out = pd.concat(out_rows, ignore_index=True)
    out["debt"] = out.debt.fillna(0)
    # "high_debt" = top quartile AMONG HH with positive debt (the within-wave 75th
    # percentile of the FULL distribution falls at zero in IFLS4 since 84 % of HH
    # didn't borrow, which makes a naive q75 flag meaningless).
    def q4_among_borrowers(s):
        pos = s[s > 0]
        if len(pos) == 0:
            return pd.Series(0, index=s.index, dtype=int)
        thr = pos.quantile(0.75)
        return ((s > 0) & (s >= thr)).astype(int)
    out["debt_q4"] = out.groupby("wave")["debt"].transform(q4_among_borrowers)
    return out


def _medical_oop_from_rn1(rn: pd.DataFrame, *, wave: str) -> pd.DataFrame:
    """Aggregate one wave's inpatient OOP costs to person-wave rows."""
    rn["oop"] = _clean_money(rn.rn19)
    # One row per inpatient-care episode; sum to person.
    rn = rn.groupby("pidlink")["oop"].sum().reset_index()
    rn["wave"] = wave
    return rn


def _high_medical_oop() -> pd.DataFrame:
    """Top-quartile of hospitalisation out-of-pocket cost, among hospitalised adults."""
    out_rows = []
    for wave in ["IFLS4", "IFLS5"]:
        p = wave_folder(RAW, wave) / "b3b_rn1.dta"
        if not p.exists():
            continue
        rn = pd.read_stata(p, convert_categoricals=False)
        if "rn19" not in rn.columns:
            continue
        out_rows.append(_medical_oop_from_rn1(rn, wave=wave))
    if not out_rows:
        return pd.DataFrame(columns=["pidlink", "wave", "med_oop", "high_med_oop"])
    out = pd.concat(out_rows, ignore_index=True)
    out["med_oop"] = out.oop.fillna(0)
    out = out.drop(columns=["oop"])
    # Top-quartile WITHIN wave AMONG those with non-zero OOP
    def q4_flag(s):
        thr = s[s > 0].quantile(0.75) if (s > 0).any() else np.inf
        return (s >= thr).astype(int)
    out["high_med_oop"] = out.groupby("wave")["med_oop"].transform(q4_flag)
    return out


def _pce_decline() -> pd.DataFrame:
    """Bottom-quartile of inter-wave PCE change for panel respondents."""
    stress = pd.read_parquet(OUT / "stressors.parquet")
    pce = stress[["pidlink", "wave", "pce"]].dropna(subset=["pce"])
    pce = pce[pce.pce > 0]
    pce_w = pce.pivot_table(index="pidlink", columns="wave", values="pce", aggfunc="mean")
    if "IFLS4" not in pce_w.columns or "IFLS5" not in pce_w.columns:
        return pd.DataFrame(columns=["pidlink", "wave", "pce_decline_q4"])
    panel = pce_w.dropna()
    # Approximate IDR inflation factor from 2007 to 2014 ≈ 1.7 (CPI rose ~70 %).
    # Real change = PCE_5 - PCE_4 * 1.7
    DEFLATOR = 1.7
    panel["pce_chg_real"] = panel["IFLS5"] - panel["IFLS4"] * DEFLATOR
    panel["pce_pct_chg"] = panel["pce_chg_real"] / (panel["IFLS4"] * DEFLATOR)
    thr = panel["pce_pct_chg"].quantile(0.25)
    panel["pce_decline_q4_flag"] = (panel["pce_pct_chg"] <= thr).astype(int)

    # Broadcast the same flag to BOTH wave-rows for each panel respondent
    panel = panel.reset_index()[["pidlink", "pce_decline_q4_flag"]]
    out = pd.concat([
        panel.assign(wave="IFLS4"),
        panel.assign(wave="IFLS5"),
    ], ignore_index=True)
    out = out.rename(columns={"pce_decline_q4_flag": "pce_decline_q4"})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    debt = _high_debt()
    print(f"debt rows: {len(debt):,};  debt_q4 share by wave:\n{debt.groupby('wave').debt_q4.mean().round(3)}")

    moop = _high_medical_oop()
    print(f"\nmed-OOP rows: {len(moop):,};  high_med_oop share by wave:\n"
          f"{moop.groupby('wave').high_med_oop.mean().round(4)}")

    pce_d = _pce_decline()
    print(f"\nPanel PCE rows: {len(pce_d):,};  pce_decline_q4 share by wave:\n"
          f"{pce_d.groupby('wave').pce_decline_q4.mean().round(3)}")

    # Merge to (pidlink, wave) skeleton from stressors.parquet
    stress = pd.read_parquet(OUT / "stressors.parquet")
    base = stress[["pidlink", "wave", "hhid"]].drop_duplicates(["pidlink", "wave"])
    out = base.merge(debt[["hhid", "wave", "debt", "debt_q4"]], on=["hhid", "wave"], how="left")
    out = out.merge(moop[["pidlink", "wave", "med_oop", "high_med_oop"]], on=["pidlink", "wave"], how="left")
    out = out.merge(pce_d[["pidlink", "wave", "pce_decline_q4"]], on=["pidlink", "wave"], how="left")

    for c in ["debt_q4", "high_med_oop", "pce_decline_q4"]:
        out[c] = out[c].fillna(0).astype(int)
    for c in ["debt", "med_oop"]:
        out[c] = out[c].fillna(0)

    out = FINANCE_DISTRESS_SHOCKS_SCHEMA.validate(out)
    out.to_parquet(OUT / "finance_distress_shocks.parquet", index=False)
    print(f"\nwrote {len(out):,} rows to {OUT/'finance_distress_shocks.parquet'}")
    print("\nFinal stressor prevalence by wave:")
    print(out.groupby("wave").agg(
        n=("pidlink", "size"),
        debt_q4_pct=("debt_q4", lambda s: 100*s.mean()),
        high_med_oop_pct=("high_med_oop", lambda s: 100*s.mean()),
        pce_decline_q4_pct=("pce_decline_q4", lambda s: 100*s.mean()),
    ).round(2))


if __name__ == "__main__":
    main()
