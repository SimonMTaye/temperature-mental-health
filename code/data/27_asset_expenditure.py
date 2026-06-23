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

CASH_TRANSFER_CARD_FIELDS = {
    "kr26": "cash_transfer_health_card",
    "kr27": "cash_transfer_dana_sehat",
    "kr27a": "cash_transfer_poverty_certificate",
    "kr27b": "cash_transfer_blt_blsm_card",
    "kr27d": "cash_transfer_bsm",
    "kr27e": "cash_transfer_jslu",
    "kr27f": "cash_transfer_disability",
    "kr27g": "cash_transfer_child_welfare",
    "kr27h": "cash_transfer_troubled_youth",
    "kr27i": "cash_transfer_social_security_card",
    "kr27j1x": "cash_transfer_bpjs_health",
    "kr27j2x": "cash_transfer_bpjs_accident",
    "kr27j3x": "cash_transfer_bpjs_retirement",
    "kr27j4x": "cash_transfer_bpjs_life",
    "kr27k": "cash_transfer_family_card",
}
CASH_TRANSFER_CARD_COLUMNS = list(CASH_TRANSFER_CARD_FIELDS.values())
CASH_TRANSFER_RECIPIENT_COLUMNS = [
    "any_cash_transfer_ksr",
    "cash_transfer_blt_blsm_card",
]


def _clean_binary_response(values: pd.Series) -> pd.Series:
    return values.map({1: 1, 3: 0, 6: 0, 8: pd.NA, 9: pd.NA}).astype("Int32")


def _collapse_binary_response(values: pd.Series) -> object:
    clean = _clean_binary_response(values)
    if clean.eq(1).any():
        return 1
    if clean.isna().any():
        return pd.NA
    return 0


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
            .apply(_collapse_binary_response)
            .reset_index()
        )
        rec.columns = ["hhid", "any_cash_transfer_ksr"]
        rec["any_cash_transfer_ksr"] = rec["any_cash_transfer_ksr"].astype("Int32")
        rows.append(rec)

    if kr is not None:
        cols = {"hhid": kr[hhid_col_name]}
        for raw_col, clean_col in CASH_TRANSFER_CARD_FIELDS.items():
            if raw_col in kr.columns:
                cols[clean_col] = _clean_binary_response(kr[raw_col])
            else:
                cols[clean_col] = pd.Series(0, index=kr.index, dtype="Int32")
        rows.append(pd.DataFrame(cols).drop_duplicates("hhid"))

    if not rows:
        return pd.DataFrame(columns=["hhid", "wave"])

    out = rows[0]
    for row in rows[1:]:
        out = out.merge(row, on="hhid", how="outer", validate="1:1")
    component_cols = ["any_cash_transfer_ksr", *CASH_TRANSFER_CARD_COLUMNS]
    for col in component_cols:
        if col not in out.columns:
            out[col] = pd.Series(0, index=out.index, dtype="Int32")
        out[col] = out[col].astype("Int32")
    out["cash_transfer_recipient"] = (
        out[CASH_TRANSFER_RECIPIENT_COLUMNS].fillna(0).max(axis=1).astype("Int32")
    )
    out["wave"] = wave
    return out


def _fill_household_binary_exposures(out: pd.DataFrame) -> pd.DataFrame:
    out = out.assign(
        vehicle_owner=lambda df: df.vehicle_owner.fillna(0).astype(int),
        urban=lambda df: df.urban.fillna(0).astype(int),
        urban_vehicle_hh=lambda df: df.urban * df.vehicle_owner,
        cash_transfer_recipient=lambda df: df.cash_transfer_recipient.fillna(0).astype(
            "Int32"
        ),
    )
    for col in CASH_TRANSFER_CARD_COLUMNS:
        out[col] = out[col].astype("Int32")
    return out


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
