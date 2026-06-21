"""Build the canonical analysis input.

This is the single downstream-facing person-wave artifact consumed by analysis,
table, and figure scripts.

Output: data/generated/30_analysis_table_input.parquet
"""

import numpy as np
import pandas as pd

from data.config import GENERATED_DATA, IDR_2007_TO_2014_INFLATOR
from data._schemas import ANALYSIS_TABLE_INPUT_SCHEMA, CURRENCY_CONVERSIONS_SCHEMA
from library.log import log

POST_SUBSIDY_DATE = pd.Timestamp("2014-11-18")
HAZE_MONTHS = {(2015, 9), (2015, 10), (2015, 11)}
CESD_FACTOR_COLUMNS = [
    "somatic",
    "depraffect",
    "posaffect",
    "somatic_z",
    "depraffect_z",
    "posaffect_z",
]


def add_ifsl4_measurements(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    wave4 = (
        df
        # Filter to rows where wave = ifls4
        .query("wave == 'IFLS4'")
        # Keep pidlink plus IFLS4 baseline worker columns
        .filter(
            items=columns + ["pidlink"],
        )
        # Add IFLS4 dummy to every column
        .add_suffix("_ifls4", axis=1)
        # Rename pidlink back to pidlink (remove suffix)
        .rename(columns={"pidlink_ifls4": "pidlink"})
    )
    return df.merge(wave4, on="pidlink", how="left", validate="m:1")


def build_core_panel() -> pd.DataFrame:
    """Merge person, CES-D, covariate, and processed temperature inputs."""
    ind = pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")
    ces = pd.read_parquet(GENERATED_DATA / "24_cesd_scores.parquet")
    ces = pd.read_parquet(GENERATED_DATA / "24_cesd_scores.parquet")
    stress = pd.read_parquet(GENERATED_DATA / "22_stressors.parquet")
    temp = pd.read_parquet(GENERATED_DATA / "26_processed_temperature_data.parquet")

    log(
        f"individuals={len(ind):,}  cesd={len(ces):,}  stressors={len(stress):,}  temp={len(temp):,}"
    )
    df = ind.merge(ces, on=["pidlink", "wave"], how="inner", validate="1:1")
    df = df.merge(
        stress.drop(columns=["hhid"], errors="ignore"),
        on=["pidlink", "wave"],
        how="left",
        validate="1:1",
    )
    log(f"after CES-D merge: {len(df):,}")

    df = df.merge(temp, on=["pidlink", "wave"], how="left", validate="1:1")
    df["interview_date"] = pd.to_datetime(df.interview_datetime).dt.normalize()
    df = df[df.age >= 15].copy()
    df = df.dropna(subset=["cesd_raw", "tmean_c"]).copy()
    log(f"core analysis sample: {len(df):,} adults")

    df["post_subsidy"] = (df.interview_date >= POST_SUBSIDY_DATE).astype(int)
    df["haze_2015"] = df.interview_date.apply(
        lambda d: int((d.year, d.month) in HAZE_MONTHS)
    )
    df["yogya_quake_catchment"] = (
        df.wave.eq("IFLS4") & df.province_code.isin([33, 34])
    ).astype(int)
    df["month_year"] = df.interview_date.dt.to_period("M").astype(str)
    df["month"] = df.interview_date.dt.month
    df["year"] = df.interview_date.dt.year
    return df


def deflate(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Deflate nominal values to real using wave-level CPI."""
    df["deflator"] = np.where(df.wave == "IFLS4", IDR_2007_TO_2014_INFLATOR, 1)
    for col in columns:
        df[f"{col}_real"] = df[col] * df.deflator
    df = df.drop(columns=["deflator"])
    return df


def main() -> None:
    GENERATED_DATA.mkdir(parents=True, exist_ok=True)

    df = build_core_panel()
    conversions = CURRENCY_CONVERSIONS_SCHEMA.validate(
        pd.read_parquet(GENERATED_DATA / "03_currency_conversions.parquet")
    )
    economic = pd.read_parquet(GENERATED_DATA / "20_economic_exposures.parquet")
    expenditure = pd.read_parquet(GENERATED_DATA / "25_expenditure_data.parquet")
    asset_expenditure = pd.read_parquet(GENERATED_DATA / "27_asset_expenditure.parquet")

    df = df.merge(
        conversions, on=["year", "month"], how="left", validate="m:1"
    )
    df = df.merge(economic, on=["pidlink", "wave"], how="left", validate="1:1")
    df = df.merge(
        expenditure,
        on=["pidlink", "wave"],
        how="left",
        validate="1:1",
    )
    df = df.merge(asset_expenditure, on=["pidlink", "wave"], how="left", validate="1:1")
    df = df.merge(
        pd.read_parquet(GENERATED_DATA / "28_sleep_duration.parquet"),
        on=["pidlink", "wave"],
        how="left",
        validate="1:1",
    )

    df = df.merge(
        pd.read_parquet(GENERATED_DATA / "21_health_bereavement_shocks.parquet"),
        on=["pidlink", "wave"],
        how="left",
        validate="1:1",
    )
    df = df.merge(
        pd.read_parquet(GENERATED_DATA / "23_finance_distress_shocks.parquet").drop(
            columns=["hhid", "debt", "med_oop"],
            errors="ignore",
        ),
        on=["pidlink", "wave"],
        how="left",
        validate="1:1",
    )

    df = (
        df.pipe(
            add_ifsl4_measurements,
            columns=[
                "urban_vehicle_hh",
                "coal_worker_hh",
                "coal_worker_individual",
                "palm_farmer_hh",
                "palm_farmer_individual",
                "fuel_share",
                "fuel_transport_share",
                "fuel_share_z",
                "fuel_transport_share_z",
                "fuel_share_quartile",
            ],
        )
        .assign(
            female=lambda df: df["sex"].eq("F").astype(int),
            cesd_z=lambda df: df.groupby("wave")["cesd_raw"].transform(
                lambda s: (s - s.mean()) / s.std()
            ),
            palm_price_wave5=lambda df: (
                df["palm_price_usd_mt"]
                .where(df["wave"] == "IFLS5")
                .groupby(df["pidlink"])
                .transform("max")
            ),
            palm_price_wave4=lambda df: (
                df["palm_price_usd_mt"]
                .where(df["wave"] == "IFLS4")
                .groupby(df["pidlink"])
                .transform("max")
            ),
            palm_price_gap=lambda df: (
                (df.palm_price_wave4 - df.palm_price_wave5) * df.palm_farmer_hh_ifls4
            ),
            palm_price_gap_z=lambda df: (
                (
                    (df.palm_price_gap - df.palm_price_gap.mean())
                    / df.palm_price_gap.std()
                )
                * df.palm_farmer_hh_ifls4
            ),
            # Deflate total expenditure and job income variables and non labor income
        )
        .pipe(
            deflate,
            columns=[
                "job_earnings_individual",
                "job_earnings_hh",
                "hh_nonlabor_income_mo",
                "expenditure_nonfood_fuel_mo",
                "expenditure_food_total_mo",
                "expenditure_nonfood_total_mo",
                "expenditure_total_mo",
            ],
        )
    )

    int_cols = [
        "job_loss_1_yr",
        "palm_farmer_individual",
        "farmer_hh",
        "palm_farmer_hh",
        "rubber_farmer_individual",
        "coffee_farmer_individual",
        "coal_worker_individual",
        "coal_worker_hh",
        "vehicle_owner",
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)

    df = df[list(ANALYSIS_TABLE_INPUT_SCHEMA.columns)]
    df = ANALYSIS_TABLE_INPUT_SCHEMA.validate(df)
    out_path = GENERATED_DATA / "30_analysis_table_input.parquet"
    df.to_parquet(out_path, index=False)
    log(f"wrote {len(df):,} rows to {out_path}")
    log(
        df.groupby("wave")
        .agg(
            n=("pidlink", "size"),
            cesd_z_mean=("cesd_z", "mean"),
            job_loss_pct=("job_loss_1_yr", lambda s: 100 * s.mean()),
            palm_farmer_hh_pct=("palm_farmer_hh", lambda s: 100 * s.mean()),
            coal_worker_hh_pct=("coal_worker_hh", lambda s: 100 * s.mean()),
        )
        .round(4),
        "DEBUG",
    )


if __name__ == "__main__":
    main()
