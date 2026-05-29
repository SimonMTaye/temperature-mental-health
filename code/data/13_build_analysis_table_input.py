"""Build the table-ready analysis input.

This is the downstream-facing person-wave artifact consumed by headline table
and figure scripts. It keeps `analysis_dataset.parquet` as the thin core panel
and materializes repeated table-prep variables in one place.

Output: data/generated/analysis_table_input.parquet
"""
from __future__ import annotations

import pandas as pd

from config import OUT
from _commodity_prices import COFFEE_PRICE, PALM_3MO_DECLINE, RUBBER_PRICE
from _schemas import ANALYSIS_TABLE_INPUT_SCHEMA


def merge_optional_sidecar(
    df: pd.DataFrame,
    path_name: str,
    fill_zero_cols: list[str],
    *,
    drop_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Left-merge an optional sidecar and fill indicator/count columns with zero."""
    path = OUT / path_name
    if path.exists():
        sidecar = pd.read_parquet(path)
        if drop_cols:
            sidecar = sidecar.drop(columns=drop_cols, errors="ignore")
        df = df.merge(sidecar, on=["pidlink", "wave"], how="left")
    for col in fill_zero_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)
    return df


def merge_cesd_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Merge CES-D factor scores produced by 04_score_cesd.py."""
    factors = pd.read_parquet(
        OUT / "cesd_scores.parquet",
        columns=[
            "pidlink",
            "wave",
            "somatic",
            "depraffect",
            "posaffect",
            "somatic_z",
            "depraffect_z",
            "posaffect_z",
        ],
    )
    return df.merge(factors, on=["pidlink", "wave"], how="left")


def add_model_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add common outcome, heat, threshold, and shock variables used by tables."""
    df = df.copy()
    df["precip_mm"] = df.precip_mm.clip(lower=0)
    df["female"] = (df.sex == "F").astype(int)
    df["cesd_z"] = df.groupby("wave")["cesd_raw"].transform(lambda s: (s - s.mean()) / s.std())

    for col in ["tmean_c", "tmax_c", "tmin_c"]:
        df[f"{col}_dev"] = df[col] - df[col].mean()
    df["heat_c_dev"] = df["tmean_c_dev"]

    df["cdd_tmax30"] = (df.tmax_c - 30.0).clip(lower=0)
    df["cdd_tmax32"] = (df.tmax_c - 32.0).clip(lower=0)
    df["cdd_tmin23"] = (df.tmin_c - 23.0).clip(lower=0)
    df["cdd_tmin24"] = (df.tmin_c - 24.0).clip(lower=0)

    intvw_ym = list(zip(df.interview_date.dt.year, df.interview_date.dt.month))
    df["palm_3mo_decline"] = pd.Series(intvw_ym, index=df.index).map(PALM_3MO_DECLINE).fillna(0.0)
    df["palm_shock"] = df.palm_farmer_hh * df.palm_3mo_decline
    df["fuel_shock"] = df.post_subsidy * df.transport_share

    rp = pd.Series(list(RUBBER_PRICE.values()))
    df["rubber_price_usd_kg"] = pd.Series(intvw_ym, index=df.index).map(RUBBER_PRICE)
    df["rubber_price_z"] = (df.rubber_price_usd_kg - rp.mean()) / rp.std()
    df["rubber_shock"] = df.rubber_farmer_individual * (-df.rubber_price_z.fillna(0)).clip(lower=0)

    cp = pd.Series(list(COFFEE_PRICE.values()))
    df["coffee_price_clb"] = pd.Series(intvw_ym, index=df.index).map(COFFEE_PRICE)
    df["coffee_price_z"] = (df.coffee_price_clb - cp.mean()) / cp.std()
    df["coffee_shock"] = df.coffee_farmer_individual * (-df.coffee_price_z.fillna(0)).clip(lower=0)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(OUT / "analysis_dataset.parquet")
    fin = pd.read_parquet(OUT / "financial_shocks.parquet")
    fin2 = pd.read_parquet(OUT / "financial_shocks_v2.parquet")

    df = df.merge(fin, on=["pidlink", "wave"], how="left")
    df = df.merge(fin2, on=["pidlink", "wave"], how="left", suffixes=("", "_v2"))
    if "palm_region_v2" in df.columns:
        df = df.drop(columns=["palm_region_v2"])
    df = merge_cesd_factors(df)

    df = merge_optional_sidecar(
        df,
        "health_bereavement_shocks.parquet",
        [
            "n_symptoms",
            "many_symptoms",
            "recent_hospitalised",
            "recent_accident_2y",
            "recently_widowed_5y",
        ],
    )
    df = merge_optional_sidecar(
        df,
        "finance_distress_shocks.parquet",
        ["debt_q4", "high_med_oop", "pce_decline_q4"],
        drop_cols=["hhid", "debt", "med_oop"],
    )

    df = add_model_variables(df)
    df = df.dropna(
        subset=[
            "job_loss_within_yr",
            "palm_farmer_individual",
            "palm_farmer_hh",
            "rubber_farmer_individual",
            "coffee_farmer_individual",
            "transport_share",
            "vehicle_owner",
        ]
    ).copy()

    int_cols = [
        "job_loss_within_yr",
        "palm_farmer_individual",
        "palm_farmer_hh",
        "rubber_farmer_individual",
        "coffee_farmer_individual",
        "vehicle_owner",
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)

    df = df[list(ANALYSIS_TABLE_INPUT_SCHEMA.columns)]
    df = ANALYSIS_TABLE_INPUT_SCHEMA.validate(df)
    df.to_parquet(OUT / "analysis_table_input.parquet", index=False)
    print(f"wrote {len(df):,} rows to {OUT/'analysis_table_input.parquet'}")
    print(
        df.groupby("wave").agg(
            n=("pidlink", "size"),
            cesd_z_mean=("cesd_z", "mean"),
            job_loss_pct=("job_loss_within_yr", lambda s: 100 * s.mean()),
            palm_shock_mean=("palm_shock", "mean"),
            fuel_shock_mean=("fuel_shock", "mean"),
        ).round(4)
    )


if __name__ == "__main__":
    main()
