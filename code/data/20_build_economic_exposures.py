"""Build job-loss, household-asset, benefit-card, and palm-price exposures.

Output: data/generated/20_economic_exposures.parquet
Row level: one record per (pidlink, wave), using the individual panel skeleton
from 01_individuals.parquet.
"""

import numpy as np
import pandas as pd

from _commodity_prices import PALM_PRICE_FULL  # noqa: E402
from _schemas import ECONOMIC_EXPOSURES_SCHEMA  # noqa: E402
from _sentinels import clean_count, clean_month, clean_year  # noqa: E402
from _stata import read_stata_df  # noqa: E402
from config import GENERATED_DATA, IFLS4_FOLDER, IFLS5_FOLDER  # noqa: E402
from log import log  # noqa: E402


PALM_PROVS = {
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    21,
    61,
    62,
    63,
    64,
}

OUTPUT_COLUMNS = [
    "pidlink",
    "wave",
    "unemployed_since_then",
    "lost_job_5y",
    "job_loss_5y",
    "involuntary_loss_5y",
    "days_since_last_loss",
    "lost_job_90d",
    "lost_job_180d",
    "lost_job_270d",
    "lost_job_1_yr",
    "lost_job_365d",
    "lost_job_540d",
    "lost_job_730d",
    "lost_job_1095d",
    "lost_job_1825d",
    "lost_job_6_months",
    "lost_job_3_months",
    "job_loss_90d",
    "job_loss_180d",
    "job_loss_270d",
    "job_loss_1_yr",
    "job_loss_365d",
    "job_loss_540d",
    "job_loss_730d",
    "job_loss_1095d",
    "job_loss_1825d",
    "job_loss_6_months",
    "job_loss_3_months",
    "job_loss_reason_code",
    "involuntary_loss_1_yr",
    "family_loss_1_yr",
    "vehicle_owner",
    "urban",
    "urban_vehicle_hh",
    "urban_vehicle_hh_ifls4",
    "cash_transfer_recipient",
    "blt_card",
    "health_card",
    "palm_region",
    "palm_price_usd_mt",
    "palm_price_z",
    "palm_shock",
]

BINARY_COLUMNS = [
    "unemployed_since_then",
    "lost_job_5y",
    "job_loss_5y",
    "involuntary_loss_5y",
    "lost_job_90d",
    "lost_job_180d",
    "lost_job_270d",
    "lost_job_1_yr",
    "lost_job_365d",
    "lost_job_540d",
    "lost_job_730d",
    "lost_job_1095d",
    "lost_job_1825d",
    "lost_job_6_months",
    "lost_job_3_months",
    "job_loss_90d",
    "job_loss_180d",
    "job_loss_270d",
    "job_loss_1_yr",
    "job_loss_365d",
    "job_loss_540d",
    "job_loss_730d",
    "job_loss_1095d",
    "job_loss_1825d",
    "job_loss_6_months",
    "job_loss_3_months",
    "involuntary_loss_1_yr",
    "family_loss_1_yr",
    "vehicle_owner",
    "urban",
    "urban_vehicle_hh",
    "cash_transfer_recipient",
    "blt_card",
    "health_card",
    "palm_region",
]

IFLS_FOLDERS = {
    "IFLS4": IFLS4_FOLDER,
    "IFLS5": IFLS5_FOLDER,
}

HHID_COLUMNS = {
    "IFLS4": "hhid07",
    "IFLS5": "hhid14",
}

LOSS_SUFFIXES = [
    "5y",
    "90d",
    "180d",
    "270d",
    "1_yr",
    "365d",
    "540d",
    "730d",
    "1095d",
    "1825d",
    "6_months",
    "3_months",
]


def _job_loss_from_df(df: pd.DataFrame, *, wave: str) -> pd.DataFrame:
    """Build individual job-loss recall fields from one IFLS work-history file."""
    out = pd.DataFrame({"pidlink": df.pidlink})
    out["lost_job_5y"] = clean_count(df.tk46c, max_real=50).fillna(0) >= 1
    # 1, 2 -> Fired, 3 -> Wage too Low, 4 -> Bad Working Env, 5 -> Refused relocation
    # 6 -> Prolonged sickness, 7, 8, 9 -> Family related (marriange, child, other) 95 -> Other
    out["job_loss_reason_code"] = pd.to_numeric(df.tk46m, errors="coerce")
    out.loc[~out.job_loss_reason_code.between(1, 95), "job_loss_reason_code"] = np.nan
    out["involuntary_loss_5y"] = out.job_loss_reason_code.isin([1, 2, 3, 4, 5, 6])
    out["last_loss_year"] = clean_year(df.tk46dy)
    out["last_loss_month"] = clean_month(df.tk46dm)
    out["wave"] = wave
    return out.drop_duplicates("pidlink")


def _unemployment_from_df(df: pd.DataFrame, *, wave: str) -> pd.DataFrame:
    """Flag respondents who did no paid work in the past week."""
    out = pd.DataFrame({"pidlink": df.pidlink})
    out["unemployed_since_then"] = df.tk01a.isin([2, 3])
    out["wave"] = wave
    return out.drop_duplicates("pidlink")


def _job_loss(wave: str) -> pd.DataFrame:
    df = read_stata_df(
        IFLS_FOLDERS[wave] / "b3a_tk4.dta",
        convert_categoricals=False,
    )
    return _job_loss_from_df(df, wave=wave)


def _unemployment(wave: str) -> pd.DataFrame:
    df = read_stata_df(
        IFLS_FOLDERS[wave] / "b3a_tk1.dta",
        convert_categoricals=False,
    )
    return _unemployment_from_df(df, wave=wave)


def _vehicle_owner_from_df(
    hr: pd.DataFrame, *, hhid_col_name: str, wave: str
) -> pd.DataFrame:
    """Flag households reporting vehicle ownership in the household roster."""
    veh = hr[hr.hrtype == "E"].copy()
    veh["vehicle_owner"] = veh.hr01 == 1
    out = (
        veh[[hhid_col_name, "vehicle_owner"]]
        .rename(columns={hhid_col_name: "hhid"})
        .drop_duplicates("hhid")
    )
    out["wave"] = wave
    return out


def _vehicle_owner(wave: str) -> pd.DataFrame:
    hr = read_stata_df(
        IFLS_FOLDERS[wave] / "b2_hr1.dta",
        convert_categoricals=False,
    )
    return _vehicle_owner_from_df(hr, hhid_col_name=HHID_COLUMNS[wave], wave=wave)


def _urban_from_df(
    screening: pd.DataFrame, *, hhid_col_name: str, wave: str
) -> pd.DataFrame:
    """Flag urban households from the wave screening file."""
    screening = screening.copy()
    screening["urban"] = screening.sc05 == 1
    out = (
        screening[[hhid_col_name, "urban"]]
        .rename(columns={hhid_col_name: "hhid"})
        .drop_duplicates("hhid")
    )
    out["wave"] = wave
    return out


def _urban(wave: str) -> pd.DataFrame:
    fname = "bk_sc1.dta" if wave == "IFLS5" else "bk_sc.dta"
    screening = read_stata_df(
        IFLS_FOLDERS[wave] / fname,
        convert_categoricals=False,
    )
    return _urban_from_df(screening, hhid_col_name=HHID_COLUMNS[wave], wave=wave)


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


def _add_loss_timing(out: pd.DataFrame) -> pd.DataFrame:
    has_date = (
        out.last_loss_year.notna()
        & (out.last_loss_year > 0)
        & out.last_loss_month.notna()
        & (out.last_loss_month > 0)
    )
    out["last_loss_date"] = pd.NaT
    if has_date.any():
        years = out.loc[has_date, "last_loss_year"].astype(int)
        years = np.where(years < 100, years + 2000, years)
        months = out.loc[has_date, "last_loss_month"].astype(int).clip(1, 12)
        out.loc[has_date, "last_loss_date"] = pd.to_datetime(
            dict(year=years, month=months, day=15),
            errors="coerce",
        )
    out["last_loss_date"] = pd.to_datetime(out.last_loss_date)
    out["days_since_last_loss"] = (out.interview_date - out.last_loss_date).dt.days
    for days in [90, 180, 270, 365, 540, 730, 1095, 1825]:
        out[f"lost_job_{days}d"] = (
            (out.days_since_last_loss >= 0) & (out.days_since_last_loss <= days)
        ).astype(int)
    out["lost_job_1_yr"] = out["lost_job_365d"]
    out["lost_job_6_months"] = (
        (out.days_since_last_loss >= 0) & (out.days_since_last_loss <= 183)
    ).astype(int)
    out["lost_job_3_months"] = (
        (out.days_since_last_loss >= 0) & (out.days_since_last_loss <= 92)
    ).astype(int)

    unemployed = out.unemployed_since_then.fillna(0).astype(bool)
    for suffix in LOSS_SUFFIXES:
        out[f"job_loss_{suffix}"] = (
            out[f"lost_job_{suffix}"].eq(1) & unemployed
        ).astype(int)
    out["involuntary_loss_1_yr"] = (
        out.job_loss_1_yr.eq(1) & out.job_loss_reason_code.isin([1, 2, 3, 4, 5, 6])
    ).astype(int)
    out["family_loss_1_yr"] = (
        out.job_loss_1_yr.eq(1) & out.job_loss_reason_code.isin([7, 8, 9])
    ).astype(int)

    return out


def _add_palm_price_exposure(out: pd.DataFrame) -> pd.DataFrame:
    out["palm_region"] = out.province_code.isin(PALM_PROVS)
    out["intvw_yr"] = out.interview_date.dt.year
    out["intvw_mo"] = out.interview_date.dt.month
    out["palm_price_usd_mt"] = out.apply(
        lambda row: PALM_PRICE_FULL.get((row.intvw_yr, row.intvw_mo), np.nan),
        axis=1,
    )
    prices = pd.Series(PALM_PRICE_FULL.values())
    out["palm_price_z"] = (out.palm_price_usd_mt - prices.mean()) / prices.std()
    out["palm_shock"] = (out.palm_region * (-out.palm_price_z.fillna(0))).clip(lower=0)
    # TODO: Check where logic for palm household comes from
    return out


def _finalize_output(out: pd.DataFrame) -> pd.DataFrame:
    out_final = out[OUTPUT_COLUMNS].copy()
    for col in BINARY_COLUMNS:
        out_final[col] = out_final[col].fillna(0).astype(int)
    return ECONOMIC_EXPOSURES_SCHEMA.validate(out_final)


def _add_urban_vehicle_baseline(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["urban_vehicle_hh"] = (
        out.urban.fillna(0).astype(int) * out.vehicle_owner.fillna(0).astype(int)
    )
    baseline = (
        out.query("wave == 'IFLS4'")[["pidlink", "urban_vehicle_hh"]]
        .drop_duplicates("pidlink")
        .rename(columns={"urban_vehicle_hh": "urban_vehicle_hh_ifls4"})
    )
    return out.merge(baseline, on="pidlink", how="left", validate="m:1")


def main() -> pd.DataFrame:
    """Build and write the 20-prefixed financial shock sidecar."""
    jl = pd.concat([_job_loss("IFLS4"), _job_loss("IFLS5")], ignore_index=True)
    unemp = pd.concat([_unemployment("IFLS4"), _unemployment("IFLS5")])
    veh = pd.concat([_vehicle_owner("IFLS4"), _vehicle_owner("IFLS5")])
    urb = pd.concat([_urban("IFLS4"), _urban("IFLS5")])
    cash = pd.concat([_cash_transfer("IFLS4"), _cash_transfer("IFLS5")])
    log(
        f"job loss rows: {len(jl):,}; unemployment: {len(unemp):,}; "
        f"vehicle: {len(veh):,}; "
        f"urban: {len(urb):,}; cash: {len(cash):,}"
    )

    out_final = (
        pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")
        .assign(
            interview_date=lambda df: pd.to_datetime(
                df.interview_datetime
            ).dt.normalize()
        )
        .loc[:, ["pidlink", "wave", "hhid", "interview_date", "province_code"]]
        .drop_duplicates(["pidlink", "wave"])
        .merge(jl, on=["pidlink", "wave"], how="left", validate="1:1")
        .merge(unemp, on=["pidlink", "wave"], how="left", validate="1:1")
        .merge(veh, on=["hhid", "wave"], how="left", validate="m:1")
        .merge(urb, on=["hhid", "wave"], how="left", validate="m:1")
        .merge(cash, on=["hhid", "wave"], how="left", validate="m:1")
        .pipe(_add_loss_timing)
        .pipe(_add_palm_price_exposure)
        .pipe(_add_urban_vehicle_baseline)
        .pipe(_finalize_output)
    )
    output_path = GENERATED_DATA / "20_economic_exposures.parquet"
    out_final.to_parquet(output_path, index=False)

    log(f"wrote {len(out_final):,} rows to {output_path}")
    log("shock prevalence by wave:", "DEBUG")
    log(
        out_final.groupby("wave")
        .agg(
            n=("pidlink", "size"),
            lost_job_5y_pct=("lost_job_5y", lambda s: 100 * s.mean()),
            unemployed_pct=("unemployed_since_then", lambda s: 100 * s.mean()),
            job_loss_yr_pct=("job_loss_1_yr", lambda s: 100 * s.mean()),
            vehicle_pct=("vehicle_owner", lambda s: 100 * s.mean()),
            urban_pct=("urban", lambda s: 100 * s.mean()),
            cash_xfer_pct=("cash_transfer_recipient", lambda s: 100 * s.mean()),
            health_card_pct=("health_card", lambda s: 100 * s.mean()),
            palm_region_pct=("palm_region", lambda s: 100 * s.mean()),
        )
        .round(2),
        "DEBUG",
    )
    return out_final


if __name__ == "__main__":
    main()
