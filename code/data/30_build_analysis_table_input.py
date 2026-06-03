"""Build the canonical analysis input.

This is the single downstream-facing person-wave artifact consumed by analysis,
table, and figure scripts.

Output: data/generated/30_analysis_table_input.parquet
"""

import pandas as pd

from config import GENERATED_DATA, IDR_2007_TO_2014_DEFLATOR
from _schemas import ANALYSIS_TABLE_INPUT_SCHEMA
from log import log

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


def ensure_cesd_factor_columns(ces: pd.DataFrame) -> pd.DataFrame:
    """Keep old CES-D score artifacts readable while 24 owns factor construction."""
    missing = [col for col in CESD_FACTOR_COLUMNS if col not in ces.columns]
    if missing:
        log(
            "warning: 24_cesd_scores.parquet is missing CES-D factor columns; "
            "rerun code/data/24_score_cesd.py when raw IFLS files are mounted",
            "WARNING",
        )
        ces = ces.copy()
        for col in missing:
            ces[col] = float("nan")
    return ces


def _pce_decline(df: pd.DataFrame) -> pd.DataFrame:
    """Bottom-quartile of inter-wave PCE change for panel respondents."""
    pce = df[["pidlink", "wave", "pce"]].dropna(subset=["pce"])
    pce = pce[pce.pce > 0]
    pce_w = pce.pivot_table(
        index="pidlink", columns="wave", values="pce", aggfunc="mean"
    )
    if "IFLS4" not in pce_w.columns or "IFLS5" not in pce_w.columns:
        return pd.DataFrame(columns=["pidlink", "wave", "pce_decline_q4"])
    panel = pce_w.dropna()
    panel["pce_chg_real"] = panel["IFLS5"] - panel["IFLS4"] * IDR_2007_TO_2014_DEFLATOR
    panel["pce_pct_chg"] = panel["pce_chg_real"] / (
        panel["IFLS4"] * IDR_2007_TO_2014_DEFLATOR
    )
    thr = panel["pce_pct_chg"].quantile(0.25)
    panel["pce_decline_q4_flag"] = (panel["pce_pct_chg"] <= thr).astype(int)

    # Broadcast the same flag to BOTH wave-rows for each panel respondent
    panel = panel.reset_index()[["pidlink", "pce_decline_q4_flag"]]
    out = pd.concat(
        [
            panel.assign(wave="IFLS4"),
            panel.assign(wave="IFLS5"),
        ],
        ignore_index=True,
    )
    out = out.rename(columns={"pce_decline_q4_flag": "pce_decline_q4"})
    return out


def build_core_panel() -> pd.DataFrame:
    """Merge person, CES-D, covariate, and processed temperature inputs."""
    ind = pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")
    ces = ensure_cesd_factor_columns(
        pd.read_parquet(GENERATED_DATA / "24_cesd_scores.parquet")
    )
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


def add_model_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add common outcome and demographic variables used by tables."""
    df = df.copy()
    df["female"] = (df.sex == "F").astype(int)
    df["cesd_z"] = df.groupby("wave")["cesd_raw"].transform(
        lambda s: (s - s.mean()) / s.std()
    )
    return df


def main() -> None:
    GENERATED_DATA.mkdir(parents=True, exist_ok=True)

    df = build_core_panel()
    economic = pd.read_parquet(GENERATED_DATA / "20_economic_exposures.parquet")
    commodity_transport = pd.read_parquet(
        GENERATED_DATA / "25_commodity_transport_exposures.parquet"
    )
    income_mechanisms = pd.read_parquet(
        GENERATED_DATA / "27_income_mechanism_inputs.parquet"
    )

    df = df.merge(economic, on=["pidlink", "wave"], how="left", validate="1:1")
    df = df.merge(
        commodity_transport,
        on=["pidlink", "wave"],
        how="left",
        suffixes=("", "_commodity_transport"),
        validate="1:1",
    )
    if "palm_region_commodity_transport" in df.columns:
        df = df.drop(columns=["palm_region_commodity_transport"])
    df = df.merge(
        income_mechanisms, on=["pidlink", "wave"], how="left", validate="1:1"
    )
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

    pce_d = _pce_decline(df)
    df = df.merge(pce_d, on=["pidlink", "wave"], how="left", validate="1:1")
    df["pce_decline_q4"] = df["pce_decline_q4"].fillna(0).astype(int)

    df = add_model_variables(df)
    df = df.dropna(
        subset=[
            "job_loss_1_yr",
            "palm_farmer_individual",
            "palm_farmer_individual_ifls4",
            "palm_farmer_hh",
            "palm_farmer_hh_ifls4",
            "rubber_farmer_individual",
            "coffee_farmer_individual",
            "coal_worker_individual",
            "coal_worker_individual_ifls4",
            "coal_worker_hh",
            "coal_worker_hh_ifls4",
            "transport_share",
            "vehicle_owner",
        ]
    ).copy()

    int_cols = [
        "job_loss_1_yr",
        "palm_farmer_individual",
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
