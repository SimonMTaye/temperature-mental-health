"""Build additional financial-distress stressors.

  high_debt        : top-quartile of HH debt within wave.
                     IFLS5 uses b2_bh.bh28  (current outstanding loan stock).
                     IFLS4 uses b2_bh.bh10  (loan amount taken out in past 12 mo — a flow).
                     Each wave's quartile is measured WITHIN that wave, so wave FE absorbs
                     the conceptual difference between stock and flow.
  high_medical_oop : top-quartile (within wave) of hospitalisation OOP cost (b3b_rn1.rn19),
                     among adults who were hospitalised in past 12 months. Zero otherwise.

Output: data/generated/finance_distress_shocks.parquet
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import GENERATED_DATA, IFLS4_FOLDER, IFLS5_FOLDER  # noqa: E402
from _sentinels import clean_money as _clean_money  # noqa: E402
from _schemas import FINANCE_DISTRESS_SHOCKS_SCHEMA  # noqa: E402
from _stata import read_stata_df  # noqa: E402
from log import log  # noqa: E402

IFLS_FOLDERS = {
    "IFLS4": IFLS4_FOLDER,
    "IFLS5": IFLS5_FOLDER,
}

HHID_COLUMNS = {
    "IFLS4": "hhid07",
    "IFLS5": "hhid14",
}


def _debt_from_bh(
    bh: pd.DataFrame, *, hhid_col_name: str, debt_col: str, wave: str
) -> pd.DataFrame:
    """Normalize one wave's debt proxy to HH-wave rows."""
    bh["debt"] = _clean_money(bh[debt_col])
    out = bh[[hhid_col_name, "debt"]].rename(columns={hhid_col_name: "hhid"})
    out["wave"] = wave
    return out.groupby(["hhid", "wave"], as_index=False).agg(debt=("debt", "sum"))


def _high_debt() -> pd.DataFrame:
    """Top-quartile (within wave) of HH debt."""
    out_rows = [
        # IFLS5: bh28 = total outstanding loan stock
        _debt_from_bh(
            read_stata_df(
                IFLS5_FOLDER / "b2_bh.dta",
                convert_categoricals=False,
            ),
            hhid_col_name=HHID_COLUMNS["IFLS5"],
            debt_col="bh28",
            wave="IFLS5",
        ),
        # IFLS4: bh10 = loan amount in past 12 months (flow). Closest available proxy.
        _debt_from_bh(
            read_stata_df(
                IFLS4_FOLDER / "b2_bh.dta",
                convert_categoricals=False,
            ),
            hhid_col_name=HHID_COLUMNS["IFLS4"],
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
        p = IFLS_FOLDERS[wave] / "b3b_rn1.dta"
        if not p.exists():
            continue
        rn = read_stata_df(p, convert_categoricals=False)
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


def _finalize_finance_distress(out: pd.DataFrame) -> pd.DataFrame:
    return out.assign(
        debt_q4=lambda df: df.debt_q4.fillna(0).astype(int),
        high_med_oop=lambda df: df.high_med_oop.fillna(0).astype(int),
        debt=lambda df: df.debt.fillna(0),
        med_oop=lambda df: df.med_oop.fillna(0),
    ).pipe(FINANCE_DISTRESS_SHOCKS_SCHEMA.validate)


def main() -> None:
    debt = _high_debt()
    log(
        f"debt rows: {len(debt):,};  debt_q4 share by wave:\n{debt.groupby('wave').debt_q4.mean().round(3)}",
        "DEBUG",
    )

    moop = _high_medical_oop()
    log(
        f"\nmed-OOP rows: {len(moop):,};  high_med_oop share by wave:\n"
        f"{moop.groupby('wave').high_med_oop.mean().round(4)}",
        "DEBUG",
    )

    # Merge to (pidlink, wave) skeleton from individuals.parquet
    out = (
        pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")
        .loc[:, ["pidlink", "wave", "hhid"]]
        .drop_duplicates(["pidlink", "wave"])
        .merge(
            debt[["hhid", "wave", "debt", "debt_q4"]],
            on=["hhid", "wave"],
            how="left",
            validate="m:1",
        )
        .merge(
            moop[["pidlink", "wave", "med_oop", "high_med_oop"]],
            on=["pidlink", "wave"],
            how="left",
            validate="1:1",
        )
        .pipe(_finalize_finance_distress)
    )
    out.to_parquet(GENERATED_DATA / "23_finance_distress_shocks.parquet", index=False)
    log(
        f"\nwrote {len(out):,} rows to {GENERATED_DATA / '23_finance_distress_shocks.parquet'}"
    )
    log("Final stressor prevalence by wave:", "DEBUG")
    log(
        out.groupby("wave")
        .agg(
            n=("pidlink", "size"),
            debt_q4_pct=("debt_q4", lambda s: 100 * s.mean()),
            high_med_oop_pct=("high_med_oop", lambda s: 100 * s.mean()),
        )
        .round(2),
        "DEBUG",
    )


if __name__ == "__main__":
    main()
