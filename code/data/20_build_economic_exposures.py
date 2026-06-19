"""Build job-loss, household-asset, benefit-card, and palm-price exposures.

Output: data/generated/20_economic_exposures.parquet
Row level: one record per (pidlink, wave), using the individual panel skeleton
from 01_individuals.parquet.
"""

import numpy as np
import pandas as pd

from data._commodity_prices import (
    PALM_PRICE_FULL,
    PALM_PROVS,
    COAL_PROVS,
    RUBBER_PROVS,
    COFFEE_PROVS,
)  # noqa: E402
from data._schemas import ECONOMIC_EXPOSURES_SCHEMA  # noqa: E402
from data._sentinels import clean_count, clean_month, clean_year, clean_money  # noqa: E402
from data._stata import read_stata_df  # noqa: E402
from data.config import (
    GENERATED_DATA,
    IFLS4_FOLDER,
    IFLS5_FOLDER,
)  # noqa: E402
from library.log import log  # noqa: E402


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


def _job_loss(wave: str) -> pd.DataFrame:
    """Build individual job-loss recall fields from one IFLS work-history file."""
    df = read_stata_df(
        IFLS_FOLDERS[wave] / "b3a_tk4.dta",
        convert_categoricals=False,
    )
    out = pd.DataFrame({"pidlink": df.pidlink})
    out["lost_job_5y"] = clean_count(df.tk46c, max_real=50).fillna(0) >= 1
    # 1, 2 -> Fired, 3 -> Wage too Low, 4 -> Bad Working Env, 5 -> Refused relocation
    # 6 -> Prolonged sickness, 7, 8, 9 -> Family related (marriange, child, other) 95 -> Other
    out["job_loss_reason_code"] = pd.to_numeric(df.tk46m, errors="coerce")
    out["job_loss_involuntary"] = out.job_loss_reason_code.isin([1, 2, 3, 4, 5]).astype(
        "Int32"
    )
    out["job_loss_family"] = out.job_loss_reason_code.isin([7, 8, 9]).astype("Int32")
    out.loc[~out.job_loss_reason_code.between(1, 95), "job_loss_reason_code"] = np.nan
    out["job_loss_sector"] = pd.to_numeric(df.tk46g, errors="coerce")
    out.loc[~out.job_loss_sector.between(1, 95), "job_loss_sector"] = np.nan
    out["job_loss_last_loss_year"] = clean_year(df.tk46dy)
    out["job_loss_last_loss_month"] = clean_month(df.tk46dm)
    out["wave"] = wave
    return out.drop_duplicates("pidlink")


def _unemployment(wave: str) -> pd.DataFrame:
    df = read_stata_df(
        IFLS_FOLDERS[wave] / "b3a_tk1.dta",
        convert_categoricals=False,
    )
    out = pd.DataFrame({"pidlink": df.pidlink})
    out["unemployed_since_then"] = df.tk01a.eq(3)
    out["wave"] = wave
    return out.drop_duplicates("pidlink")


def current_employment(wave: str) -> pd.DataFrame:
    """Preserve current job, self-employment, and labor-earnings fields."""
    df = read_stata_df(
        IFLS_FOLDERS[wave] / "b3a_tk2.dta",
        convert_categoricals=False,
    )
    out = pd.DataFrame(
        {
            "pidlink": df.pidlink,
            "hhid": df[HHID_COLUMNS[wave]],
            "wave": wave,
        }
    )
    out["current_job_sector"] = pd.to_numeric(df.tk19ab, errors="coerce")
    # Replace out of bounds (> 99 or < 1) with missing
    out["current_job_sector"] = out["current_job_sector"].where(
        out["current_job_sector"].between(1, 99)
    )
    out["agricultural"] = out["current_job_sector"].eq(1).fillna(0)
    out["mining"] = out["current_job_sector"].eq(2).fillna(0)
    labor_income_parts = pd.concat(
        [
            clean_money(df.tk25a1),
            clean_money(df.tk25b1),
            clean_money(df.tk26a1),
            clean_money(df.tk26b1),
        ],
        axis=1,
    )
    out["job_earnings_individual"] = labor_income_parts.sum(axis=1, min_count=1)
    out["job_earnings_hh"] = out.groupby(["hhid", "wave"])[
        "job_earnings_individual"
    ].transform("sum")

    # tk24a:
    ### 1:Self-employed |      4,176       17.06       17.06
    ### 2:Self employed with  unpaid family/tem |      4,521       18.47       35.52
    ### 3:Self-employed with employees/permanen |        497        2.03       37.55
    ### 6:Unpaid family worker

    out["job_self_employed"] = df.tk24a.isin([1, 2, 3, 6])
    return out.drop(columns=["hhid"]).drop_duplicates("pidlink")


def _add_loss_timing(out: pd.DataFrame) -> pd.DataFrame:
    out["unemployed_since_then"] = out.unemployed_since_then.fillna(0).astype(int)
    out["lost_job_5y"] = out.lost_job_5y.fillna(0).astype(int)
    out["job_loss_involuntary"] = out.job_loss_involuntary.fillna(0).astype(int)
    out["job_loss_family"] = out.job_loss_family.fillna(0).astype(int)
    has_date = (
        out.job_loss_last_loss_year.notna()
        & (out.job_loss_last_loss_year > 0)
        & out.job_loss_last_loss_month.notna()
        & (out.job_loss_last_loss_month > 0)
    )
    out["last_loss_date"] = pd.NaT
    if has_date.any():
        years = out.loc[has_date, "job_loss_last_loss_year"].astype(int)
        years = np.where(years < 100, years + 2000, years)
        months = out.loc[has_date, "job_loss_last_loss_month"].astype(int).clip(1, 12)
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
    out["involuntary_loss_5y"] = (
        out.lost_job_5y.eq(1) & out.job_loss_reason_code.isin([1, 2, 3, 4, 5, 6])
    ).astype(int)
    out["family_loss_1_yr"] = (
        out.job_loss_1_yr.eq(1) & out.job_loss_reason_code.isin([7, 8, 9])
    ).astype(int)

    return out


def _add_palm_price_exposure(out: pd.DataFrame) -> pd.DataFrame:
    out["palm_region"] = out.province_code.isin(PALM_PROVS).astype(int)
    agricultural = out.agricultural.fillna(0).astype(bool)
    mining = out.mining.fillna(0).astype(bool)
    out["palm_farmer_individual"] = (
        agricultural & out.palm_region.astype(bool)
    ).astype(int)
    out["farmer_hh"] = (
        agricultural.groupby([out["hhid"], out["wave"]]).transform("max").astype(int)
    )
    out["palm_farmer_hh"] = (
        out.groupby(["hhid", "wave"])["palm_farmer_individual"]
        .transform("max")
        .astype(int)
    )
    out["palm_price_usd_mt"] = out.apply(
        lambda row: PALM_PRICE_FULL.get(
            (row.interview_date.year, row.interview_date.month), np.nan
        ),
        axis=1,
    )
    out["rubber_region"] = out.province_code.isin(RUBBER_PROVS).astype(int)
    out["rubber_farmer_individual"] = (
        agricultural & out.rubber_region.astype(bool)
    ).astype(int)
    out["coffee_region"] = out.province_code.isin(COFFEE_PROVS).astype(int)
    out["coffee_farmer_individual"] = (
        agricultural & out.coffee_region.astype(bool)
    ).astype(int)
    out["coal_region"] = out.province_code.isin(COAL_PROVS).astype(int)
    out["coal_worker_individual"] = (mining & out.coal_region.astype(bool)).astype(int)
    out["coal_worker_hh"] = (
        out.groupby(["hhid", "wave"])["coal_worker_individual"]
        .transform("max")
        .astype(int)
    )
    return out


def _finalize_output(out: pd.DataFrame) -> pd.DataFrame:
    out_final = out[list(ECONOMIC_EXPOSURES_SCHEMA.columns)].copy()
    return ECONOMIC_EXPOSURES_SCHEMA.validate(out_final)


def main() -> pd.DataFrame:
    """Build and write the 20-prefixed financial shock sidecar."""
    jl = pd.concat([_job_loss("IFLS4"), _job_loss("IFLS5")], ignore_index=True)
    unemp = pd.concat([_unemployment("IFLS4"), _unemployment("IFLS5")])
    employment = pd.concat([current_employment("IFLS4"), current_employment("IFLS5")])

    log(
        f"job loss rows: {len(jl):,}; unemployment: {len(unemp):,}; "
        f"current job sector: {len(employment):,}; "
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
        .merge(employment, on=["pidlink", "wave"], how="left", validate="1:1")
        .pipe(_add_loss_timing)
        .pipe(_add_palm_price_exposure)
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
            palm_region_pct=("palm_region", lambda s: 100 * s.mean()),
        )
        .round(2),
        "DEBUG",
    )
    return out_final


if __name__ == "__main__":
    main()
