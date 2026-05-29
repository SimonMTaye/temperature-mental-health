"""Extract individual-level IFLS4 + IFLS5 records with interview date + admin codes.

For each wave, returns one row per (pidlink, wave) with:
  pidlink, hhid, wave, interview_date, province_code, kabupaten_code, kecamatan_code

province_code  = BPS 2-digit
kabupaten_code = BPS 4-digit  (province*100 + within-province kabupaten)
kecamatan_code = BPS 7-digit  (kabupaten*1000 + within-kabupaten kecamatan)

Output: data/generated/individuals.parquet
"""

import pandas as pd

from config import GENERATED_DATA, RAW_IFLS_EXTRACTED
from _ifls_wave import ScreeningColumns, WaveConfig, wave_config, wave_folder
from _schemas import INDIVIDUALS_SCHEMA
from _stata import read_stata_df


def admin_codes_from_screening(
    screening: pd.DataFrame, *, hhid_col: str, admin_cols: ScreeningColumns
) -> pd.DataFrame:
    """Extract household geography from a wave's screening file.

    The IFLS screening books store province, within-province kabupaten, and
    within-kabupaten kecamatan codes under wave-specific raw column names. This
    normalizes those names and returns one row per household with stable BPS-style
    `province_code`, `kabupaten_code`, and `kecamatan_code` columns.
    """
    screening = screening[
        [
            hhid_col,
            admin_cols.province,
            admin_cols.kabupaten_in_province,
            admin_cols.kecamatan_in_kabupaten,
        ]
    ].rename(
        columns={
            hhid_col: "hhid",
            admin_cols.province: "province",
            admin_cols.kabupaten_in_province: "kabupaten_in_province",
            admin_cols.kecamatan_in_kabupaten: "kecamatan_in_kabupaten",
        }
    )
    screening = screening.dropna(
        subset=["province", "kabupaten_in_province", "kecamatan_in_kabupaten"]
    ).drop_duplicates("hhid")
    screening["province"] = screening.province.astype(int)
    screening["kabupaten_in_province"] = screening.kabupaten_in_province.astype(int)
    screening["kecamatan_in_kabupaten"] = screening.kecamatan_in_kabupaten.astype(int)
    screening["province_code"] = screening.province
    screening["kabupaten_code"] = (
        screening.province * 100 + screening.kabupaten_in_province
    )
    screening["kecamatan_code"] = (
        screening.kabupaten_code * 1000 + screening.kecamatan_in_kabupaten
    )
    return screening[["hhid", "province_code", "kabupaten_code", "kecamatan_code"]]


def _individuals_from_frames(
    dates: pd.DataFrame,
    screening: pd.DataFrame,
    wave_config: WaveConfig,
) -> pd.DataFrame:
    """Build one row per person-wave with interview date and household geography.

    `dates` is the wave-specific interview-date frame and `screening` is the
    household screening frame. `wave_config` supplies the raw file metadata:
    household ID column, date columns, year transformation, and screening-code
    columns. The result is keyed by `pidlink` and `wave`, with household ID,
    interview date, and normalized admin codes.
    """
    # TODO: Check if we have hour
    date_cols = wave_config.interview_date_cols
    dates = dates[
        (dates[date_cols.year] > 0)
        & (dates[date_cols.month] > 0)
        & (dates[date_cols.day] > 0)
    ].copy()
    dates["interview_date"] = pd.to_datetime(
        dict(
            year=wave_config.year_transform(dates[date_cols.year]).astype(int),
            month=dates[date_cols.month].astype(int),
            day=dates[date_cols.day].astype(int),
        ),
        errors="coerce",
    )
    dates = dates.dropna(subset=["interview_date"])
    # TODO: Simply dropping seems crazy
    dates = dates.sort_values(["pidlink", "interview_date"]).drop_duplicates(
        "pidlink", keep="first"
    )
    dates = dates[["pidlink", wave_config.hhid_col, "interview_date"]].rename(
        columns={wave_config.hhid_col: "hhid"}
    )

    admin = admin_codes_from_screening(
        screening,
        hhid_col=wave_config.hhid_col,
        admin_cols=wave_config.screening_cols,
    )
    out = dates.merge(admin, on="hhid", how="inner")
    out["wave"] = wave_config.wave
    return INDIVIDUALS_SCHEMA.validate(
        out[
            [
                "pidlink",
                "hhid",
                "interview_date",
                "province_code",
                "kabupaten_code",
                "kecamatan_code",
                "wave",
            ]
        ]
    )


def _ifls5_individuals() -> pd.DataFrame:
    # Interview date sits in b3a_time (one row per book-3A interview attempt)
    cfg = wave_config("IFLS5")
    folder = wave_folder(RAW_IFLS_EXTRACTED, "IFLS5")
    dates = read_stata_df(folder / cfg.dates_file, convert_categoricals=False)
    screening = read_stata_df(folder / cfg.screening_file, convert_categoricals=False)
    return _individuals_from_frames(
        dates,
        screening,
        cfg,
    )


def _ifls4_individuals() -> pd.DataFrame:
    # b3a_cov has ivwday1/ivwmth1/ivwyr1 (year is 2-digit: 7=2007, 8=2008)
    cfg = wave_config("IFLS4")
    folder = wave_folder(RAW_IFLS_EXTRACTED, "IFLS4")
    dates = read_stata_df(folder / cfg.dates_file, convert_categoricals=False)
    screening = read_stata_df(folder / cfg.screening_file, convert_categoricals=False)
    return _individuals_from_frames(
        dates,
        screening,
        cfg,
    )


def main() -> None:
    parts = [_ifls4_individuals(), _ifls5_individuals()]
    out = pd.concat(parts, ignore_index=True)
    GENERATED_DATA.mkdir(parents=True, exist_ok=True)
    out.to_parquet(GENERATED_DATA / "01_individuals.parquet", index=False)
    print(
        f"wrote {len(out):,} individual-wave rows to {GENERATED_DATA / 'individuals.parquet'}"
    )
    print(
        out.groupby("wave").agg(
            n=("pidlink", "size"),
            date_min=("interview_date", "min"),
            date_max=("interview_date", "max"),
            n_provinces=("province_code", "nunique"),
            n_kabupaten=("kabupaten_code", "nunique"),
            n_kecamatan=("kecamatan_code", "nunique"),
        )
    )


if __name__ == "__main__":
    main()
