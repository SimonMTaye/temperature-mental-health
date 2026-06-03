"""Build additional stressor variables — health-related and bereavement-related.

Adds, per (pidlink, wave):
  recent_acute_symptom    1 if respondent reported ANY symptom in the past 4 weeks (b3b_ma2.ma01==1)
  recent_hospitalised     1 if hospitalised in past 12 months (b3b_rn2.rn01==1)
  recent_accident_2y      1 if had an accident with treatment in past 2 years
                          (b3b_ma1.ma15==1 AND ma16y within 2 years of interview year)
  recently_widowed_5y     1 if any past marriage ended in widowhood (kw11b==2)
                          AND ended within 5 years of interview (kw18y, kw18m)

Output: data/generated/21_health_bereavement_shocks.parquet
"""

import numpy as np
import pandas as pd

from config import GENERATED_DATA, IFLS4_FOLDER, IFLS5_FOLDER
from _schemas import HEALTH_BEREAVEMENT_SHOCKS_SCHEMA
from _stata import read_stata_df
from log import log

IFLS_FOLDERS = {
    "IFLS4": IFLS4_FOLDER,
    "IFLS5": IFLS5_FOLDER,
}


def _acute_symptom_from_df(df: pd.DataFrame, *, wave: str) -> pd.DataFrame:
    """Count of distinct symptoms reported in the past 4 weeks. Most adults have ≥1
    so we use the COUNT (mean ≈ 3) and a 'many symptoms' (top quartile, ≥ 5) flag."""
    df["had_symptom"] = (df.ma01 == 1).astype(int)
    counts = (
        df.groupby("pidlink")["had_symptom"].sum().rename("n_symptoms").reset_index()
    )
    counts["many_symptoms"] = (counts.n_symptoms >= 5).astype(int)
    counts["wave"] = wave
    return counts


def _acute_symptom(wave: str) -> pd.DataFrame:
    p = IFLS_FOLDERS[wave] / "b3b_ma2.dta"
    if not p.exists():
        return pd.DataFrame(columns=["pidlink", "wave", "n_symptoms", "many_symptoms"])
    return _acute_symptom_from_df(
        read_stata_df(p, convert_categoricals=False), wave=wave
    )


def _hospitalised_from_df(df: pd.DataFrame, *, wave: str) -> pd.DataFrame:
    df["was_hosp"] = (df.rn01 == 1).astype(int)
    # The dataset has one row per inpatient-care type; collapse to person.
    out = (
        df.groupby("pidlink")["was_hosp"]
        .max()
        .rename("recent_hospitalised")
        .reset_index()
    )
    out["wave"] = wave
    return out


def _hospitalised(wave: str) -> pd.DataFrame:
    p = IFLS_FOLDERS[wave] / "b3b_rn2.dta"
    if not p.exists():
        return pd.DataFrame(columns=["pidlink", "wave", "recent_hospitalised"])
    return _hospitalised_from_df(
        read_stata_df(p, convert_categoricals=False), wave=wave
    )


def _accident_2y_from_df(
    df: pd.DataFrame, *, wave: str, interview_dates: pd.DataFrame
) -> pd.DataFrame:
    if "ma15" not in df.columns:
        return pd.DataFrame(columns=["pidlink", "wave", "recent_accident_2y"])
    df = df[["pidlink", "ma15", "ma16mth", "ma16yr"]].copy()
    df = df.merge(interview_dates, on="pidlink", how="left", validate="m:1")
    has_acc = df.ma15 == 1
    yr = pd.to_numeric(df.ma16yr, errors="coerce")
    yr = np.where(yr < 100, yr + 2000, yr)
    mo = pd.to_numeric(df.ma16mth, errors="coerce").clip(1, 12)
    acc_date = pd.to_datetime(
        dict(
            year=pd.Series(yr, index=df.index).fillna(0).astype(int).clip(1900, 2025),
            month=mo.fillna(1).astype(int),
            day=15,
        ),
        errors="coerce",
    )
    days = (df.interview_date - acc_date).dt.days
    df["accident_within_2y"] = (
        has_acc & days.between(0, 730, inclusive="both")
    ).astype(int)
    out = (
        df.groupby("pidlink")["accident_within_2y"]
        .max()
        .rename("recent_accident_2y")
        .reset_index()
    )
    out["wave"] = wave
    return out


def _accident_2y(wave: str, interview_dates: pd.DataFrame) -> pd.DataFrame:
    p = IFLS_FOLDERS[wave] / "b3b_ma1.dta"
    if not p.exists():
        return pd.DataFrame(columns=["pidlink", "wave", "recent_accident_2y"])
    return _accident_2y_from_df(
        read_stata_df(p, convert_categoricals=False),
        wave=wave,
        interview_dates=interview_dates,
    )


def _widowed_5y_from_df(
    df: pd.DataFrame, *, wave: str, interview_dates: pd.DataFrame
) -> pd.DataFrame:
    if "kw11b" not in df.columns:
        return pd.DataFrame(columns=["pidlink", "wave", "recently_widowed_5y"])
    df = df[["pidlink", "kw11b", "kw18mth", "kw18yr"]].copy()
    df = df.merge(interview_dates, on="pidlink", how="left", validate="m:1")
    # IFLS5 codes: 8 = Widow/Widower, 7 = Divorced, 6 = Separated.
    is_widow = df.kw11b == 8
    yr = pd.to_numeric(df.kw18yr, errors="coerce")
    yr = np.where(yr < 100, yr + 2000, yr)
    mo = pd.to_numeric(df.kw18mth, errors="coerce").clip(1, 12)
    end_date = pd.to_datetime(
        dict(
            year=pd.Series(yr, index=df.index).fillna(0).astype(int).clip(1900, 2025),
            month=mo.fillna(1).astype(int),
            day=15,
        ),
        errors="coerce",
    )
    days = (df.interview_date - end_date).dt.days
    df["widow_5y"] = (is_widow & days.between(0, 365 * 5, inclusive="both")).astype(int)
    out = (
        df.groupby("pidlink")["widow_5y"]
        .max()
        .rename("recently_widowed_5y")
        .reset_index()
    )
    out["wave"] = wave
    return out


def _widowed_5y(wave: str, interview_dates: pd.DataFrame) -> pd.DataFrame:
    p = IFLS_FOLDERS[wave] / "b3a_kw3.dta"
    if not p.exists():
        return pd.DataFrame(columns=["pidlink", "wave", "recently_widowed_5y"])
    return _widowed_5y_from_df(
        read_stata_df(p, convert_categoricals=False),
        wave=wave,
        interview_dates=interview_dates,
    )


def _health_wave(wave: str, ind: pd.DataFrame) -> pd.DataFrame:
    interview_dates = ind.loc[
        ind.wave == wave, ["pidlink", "interview_date"]
    ].drop_duplicates("pidlink")
    return (
        _acute_symptom(wave)
        .merge(
            _hospitalised(wave),
            on=["pidlink", "wave"],
            how="outer",
            validate="1:1",
        )
        .merge(
            _accident_2y(wave, interview_dates),
            on=["pidlink", "wave"],
            how="outer",
            validate="1:1",
        )
        .merge(
            _widowed_5y(wave, interview_dates),
            on=["pidlink", "wave"],
            how="outer",
            validate="1:1",
        )
    )


def main() -> None:
    GENERATED_DATA.mkdir(parents=True, exist_ok=True)

    ind = (
        pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")
        .assign(
            interview_date=lambda df: pd.to_datetime(
                df.interview_datetime
            ).dt.normalize()
        )
        .loc[:, ["pidlink", "wave", "interview_date"]]
    )
    out = (
        pd.concat([_health_wave(wave, ind) for wave in ["IFLS4", "IFLS5"]])
        .assign(
            n_symptoms=lambda df: df.n_symptoms.fillna(0).astype(int),
            many_symptoms=lambda df: df.many_symptoms.fillna(0).astype(int),
            recent_hospitalised=lambda df: df.recent_hospitalised.fillna(0).astype(int),
            recent_accident_2y=lambda df: df.recent_accident_2y.fillna(0).astype(int),
            recently_widowed_5y=lambda df: df.recently_widowed_5y.fillna(0).astype(int),
        )
        .pipe(HEALTH_BEREAVEMENT_SHOCKS_SCHEMA.validate)
    )
    out_path = GENERATED_DATA / "21_health_bereavement_shocks.parquet"
    out.to_parquet(out_path, index=False)

    log(f"wrote {len(out):,} rows to {out_path}")
    log("shock prevalence by wave (% of adults):", "DEBUG")
    log(
        out.groupby("wave")
        .agg(
            n=("pidlink", "size"),
            n_symptoms_mean=("n_symptoms", "mean"),
            many_symptoms_pct=("many_symptoms", lambda s: 100 * s.mean()),
            recent_hospitalised_pct=("recent_hospitalised", lambda s: 100 * s.mean()),
            recent_accident_2y_pct=("recent_accident_2y", lambda s: 100 * s.mean()),
            recently_widowed_5y_pct=("recently_widowed_5y", lambda s: 100 * s.mean()),
        )
        .round(3),
        "DEBUG",
    )


if __name__ == "__main__":
    main()
