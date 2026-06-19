"""Build household asset, benefit-card, and nonlabor income inputs.

Output: data/generated/27_asset_expenditure.parquet
Row level: one person-wave record, keyed by pidlink + wave.
"""

import pandas as pd

from data._schemas import ASSET_EXPENDITURE_SCHEMA
from data._sentinels import clean_money
from data._stata import read_stata_df
from data.config import (
    GENERATED_DATA,
    IFLS4_FOLDER,
    IFLS5_FOLDER,
)
from library.log import log


IFLS_FOLDERS = {
    "IFLS4": IFLS4_FOLDER,
    "IFLS5": IFLS5_FOLDER,
}

HHID_COLUMNS = {
    "IFLS4": "hhid07",
    "IFLS5": "hhid14",
}


def _farm_profit(wave: str) -> pd.DataFrame:
    hhid_col = HHID_COLUMNS[wave]
    df = read_stata_df(IFLS_FOLDERS[wave] / "b2_ut1.dta", convert_categoricals=False)[
        [hhid_col, "ut09"]
    ].copy()
    return (
        df.assign(hhid=df[hhid_col], annual_profit=clean_money(df.ut09), wave=wave)
        .groupby(["hhid", "wave"], as_index=False)
        .agg(annual_profit=("annual_profit", "sum"))
    )


def _nonfarm_profit(wave: str) -> pd.DataFrame:
    hhid_col = HHID_COLUMNS[wave]
    df = read_stata_df(IFLS_FOLDERS[wave] / "b2_nt2.dta", convert_categoricals=False)[
        [hhid_col, "nt09", "nt26"]
    ].copy()
    return (
        df.assign(
            hhid=df[hhid_col],
            business_profit=clean_money(df.nt09),
            rental_income=clean_money(df.nt26),
            wave=wave,
        )
        .assign(
            annual_profit=lambda x: (
                x[["business_profit", "rental_income"]]
                .sum(axis=1, min_count=1)
                .fillna(0)
            )
        )
        .groupby(["hhid", "wave"], as_index=False)
        .agg(annual_profit=("annual_profit", "sum"))
    )


def _cash_transfer(wave: str) -> pd.DataFrame:
    folder = IFLS_FOLDERS[wave]
    ksr_path = folder / "b1_ksr1.dta"
    kr_path = folder / "b2_kr.dta"
    return _cash_transfer_from_frames(
        wave=wave,
        hhid_col_name=HHID_COLUMNS[wave],
        ksr=read_stata_df(ksr_path, convert_categoricals=False)
        if ksr_path.exists()
        else None,
        kr=read_stata_df(kr_path, convert_categoricals=False)
        if kr_path.exists()
        else None,
    )


def _cash_transfer_from_frames(
    *,
    wave: str,
    hhid_col_name: str,
    ksr: pd.DataFrame | None,
    kr: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine KSR and KR benefit/card indicators to household-wave rows."""
    rows = []
    if ksr is not None and "ksr17" in ksr.columns:
        rec = (
            ksr.groupby(hhid_col_name)["ksr17"]
            .apply(lambda x: int((x == 1).any()))
            .reset_index()
        )
        rec.columns = ["hhid", "any_cash_transfer_ksr"]
        rows.append(rec)

    if kr is not None:
        cols = {"hhid": kr[hhid_col_name]}
        if "kr27b" in kr.columns:
            cols["blt_card"] = kr.kr27b == 1
        if "kr26" in kr.columns:
            cols["health_card"] = kr.kr26 == 1
        rows.append(pd.DataFrame(cols).drop_duplicates("hhid"))

    if not rows:
        return pd.DataFrame(columns=["hhid", "wave"])

    out = rows[0]
    for row in rows[1:]:
        out = out.merge(row, on="hhid", how="outer", validate="1:1")
    for col in ["any_cash_transfer_ksr", "blt_card", "health_card"]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)
    out["cash_transfer_recipient"] = (
        out.filter(items=["any_cash_transfer_ksr", "blt_card"])
        .max(axis=1)
        .fillna(0)
        .astype(int)
    )
    out["wave"] = wave
    return out


def _fill_household_binary_exposures(out: pd.DataFrame) -> pd.DataFrame:
    return out.assign(
        vehicle_owner=lambda df: df.vehicle_owner.fillna(0).astype(int),
        urban=lambda df: df.urban.fillna(0).astype(int),
        urban_vehicle_hh=lambda df: df.urban * df.vehicle_owner,
        cash_transfer_recipient=lambda df: df.cash_transfer_recipient.fillna(0).astype(
            int
        ),
        blt_card=lambda df: df.blt_card.fillna(0).astype(int),
        health_card=lambda df: df.health_card.fillna(0).astype(int),
    )


def _vehicle_owner(wave: str) -> pd.DataFrame:
    hr = read_stata_df(
        IFLS_FOLDERS[wave] / "b2_hr1.dta",
        convert_categoricals=False,
    )
    hhid_col_name = HHID_COLUMNS[wave]
    veh = hr[hr.hrtype == "E"].copy()
    veh["vehicle_owner"] = veh.hr01 == 1
    out = (
        veh[[hhid_col_name, "vehicle_owner"]]
        .rename(columns={hhid_col_name: "hhid"})
        .drop_duplicates("hhid")
    )
    out["wave"] = wave
    return out


def _urban(wave: str) -> pd.DataFrame:
    """Flag urban households from the wave screening file."""
    fname = "bk_sc1.dta" if wave == "IFLS5" else "bk_sc.dta"
    screening = read_stata_df(
        IFLS_FOLDERS[wave] / fname,
        convert_categoricals=False,
    )
    hhid_col_name = HHID_COLUMNS[wave]
    screening = screening.copy()
    screening["urban"] = screening.sc05 == 1
    out = (
        screening[[hhid_col_name, "urban"]]
        .rename(columns={hhid_col_name: "hhid"})
        .drop_duplicates("hhid")
    )
    out["wave"] = wave
    return out


def _nonlabor_income(wave: str) -> pd.DataFrame:
    out = pd.concat([_farm_profit(wave), _nonfarm_profit(wave)], ignore_index=True)
    out = out.groupby(["hhid", "wave"], as_index=False)["annual_profit"].sum()
    out["hh_nonlabor_income_mo"] = out.annual_profit / 12.0
    return out[["hhid", "wave", "hh_nonlabor_income_mo"]]


def build_asset_expenditure_data() -> pd.DataFrame:
    skeleton = pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")[
        ["pidlink", "hhid", "wave"]
    ].drop_duplicates(["pidlink", "wave"])
    nonlabor = pd.concat(
        [_nonlabor_income("IFLS4"), _nonlabor_income("IFLS5")], ignore_index=True
    )

    veh = pd.concat([_vehicle_owner("IFLS4"), _vehicle_owner("IFLS5")])
    urb = pd.concat([_urban("IFLS4"), _urban("IFLS5")])
    cash = pd.concat([_cash_transfer("IFLS4"), _cash_transfer("IFLS5")])
    out = (
        skeleton.merge(nonlabor, on=["hhid", "wave"], how="left", validate="m:1")
        .merge(cash, on=["hhid", "wave"], how="left", validate="m:1")
        .merge(veh, on=["hhid", "wave"], how="left", validate="m:1")
        .merge(urb, on=["hhid", "wave"], how="left", validate="m:1")
        .pipe(_fill_household_binary_exposures)
    )
    out["hh_nonlabor_income_mo"] = out.hh_nonlabor_income_mo.fillna(0)
    out = out[list(ASSET_EXPENDITURE_SCHEMA.columns)]
    return ASSET_EXPENDITURE_SCHEMA.validate(out)


def main() -> None:

    out = build_asset_expenditure_data()
    output_path = GENERATED_DATA / "27_asset_expenditure.parquet"
    out.to_parquet(output_path, index=False)
    log(f"wrote {len(out):,} rows to {output_path}")


if __name__ == "__main__":
    main()
