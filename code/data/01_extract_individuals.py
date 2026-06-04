"""
Extract individual-level IFLS4 + IFLS5 panel data with geographic identifiers

The data serves as a panel to which we will merge other variables and use to fetch
temperature data

Output: data/generated/01_individuals.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import GENERATED_DATA, RAW_IFLS_EXTRACTED
from _stata import read_stata_df
from _schemas import INDIVIDUALS_SCHEMA
from log import log

# Below are dicitonaries that store variable names for easier / cleaner code below
IFLS4 = {
    "wave": "IFLS4",
    "file": Path(RAW_IFLS_EXTRACTED) / "IFLS4" / "hh07" / "bk_sc.dta",
    "geo_columns": {
        "hhid": "hhid07",
        "province_code": "sc010707",
        "kabupaten_code": "sc020707",
        "kecamatan_code": "sc030707",
    },
    "geo_columns_00": {
        "province_code_00": "sc010700",
        "kabupaten_code_00": "sc020700",
        "kecamatan_code_00": "sc030700",
    },
}

IFLS5 = {
    "wave": "IFLS5",
    "file": Path(RAW_IFLS_EXTRACTED) / "IFLS5" / "hh14" / "bk_sc1.dta",
    "geo_columns": {
        "hhid": "hhid14",
        "province_code": "sc01_14_14",
        "kabupaten_code": "sc02_14_14",
        "kecamatan_code": "sc03_14_14",
    },
}

WAVE_GEO_CONFIGS: dict[str, dict] = {
    "IFLS5": IFLS5,
    "IFLS4": IFLS4,
}


def parse_geo_codes_ifls5() -> pd.DataFrame:
    """
    Extract household geography from a wave's screening file.
    Rename and standardize

    """
    cfg = WAVE_GEO_CONFIGS["IFLS5"]
    screening_dataset = read_stata_df(
        cfg["file"],
        convert_categoricals=False,
    )
    # Reverse the dict so mapping works
    rename_dict = {v: k for k, v in cfg["geo_columns"].items()}
    screening_dataset = screening_dataset.rename(columns=rename_dict)
    screening_dataset = screening_dataset[cfg["geo_columns"].keys()].copy()

    # We want one per household, not per interviewee
    screening_dataset["province_code"] = screening_dataset.province_code.astype(int)
    screening_dataset["kabupaten_code"] = screening_dataset.kabupaten_code.astype(int)
    screening_dataset["kecamatan_code"] = screening_dataset.kecamatan_code.astype(int)
    # Generate combined code for easy matching with GAMD boundary data.
    # The code is province (2 digits) + kabupaten (2 digits) + kecamatan (3 digits)
    screening_dataset["gadm_fullcode"] = (
        screening_dataset.province_code.astype(str).str.zfill(2)
        + screening_dataset.kabupaten_code.astype(str).str.zfill(2)
        + screening_dataset.kecamatan_code.astype(str).str.zfill(3)
    ).astype(int)
    screening_dataset["wave"] = "IFLS5"
    # Below is an indicator column to flag remapping complication when converting IFLS4 Admin codes to 5
    # Not relevant for IFLS5 so set to 0
    screening_dataset["multiple_kec_remap"] = 0
    return screening_dataset[
        [
            "hhid",
            "province_code",
            "kabupaten_code",
            "gadm_fullcode",
            "kecamatan_code",
            "wave",
            "multiple_kec_remap",
        ]
    ]


def generate_mapping_data(original_col: str) -> pd.DataFrame:
    mapping = read_stata_df(
        Path(RAW_IFLS_EXTRACTED)
        / "IFLS5"
        / "IFLS5_all_doc"
        / "IFLS5_BPS_2014_codes"
        / "kec_9899000714.dta"
    )
    # Do the same for 2000 codes
    mapping = mapping[[original_col, "kecid14"]].copy()
    mapping = mapping.dropna(subset=[original_col, "kecid14"])
    mapping = mapping.drop_duplicates(subset=[original_col, "kecid14"])
    mapping[original_col] = mapping[original_col].astype(int)
    mapping["kecid14"] = mapping.kecid14.astype(int)
    mapping = (
        mapping.groupby(original_col)
        .agg(
            gadm_fullcode_2014=("kecid14", lambda x: ",".join(x.astype(str))),
        )
        .reset_index()
    )
    return mapping


def parse_geo_codes_ifls4() -> pd.DataFrame:
    """
    Extract household geography from a wave's screening file.
    Rename and standardize

    Does additional work to convert IFLS4 2007 BPS codes to 2014 BPS codes compatible with GADM boundaries

    """
    cfg = WAVE_GEO_CONFIGS["IFLS4"]
    screening_dataset = read_stata_df(
        cfg["file"],
        convert_categoricals=False,
    )
    # Reverse the dict so mapping works
    rename_dict = {v: k for k, v in cfg["geo_columns"].items()}
    screening_dataset = screening_dataset.rename(columns=rename_dict)
    # Rename 2000 codes as well since mapping dict doesn't have all the 2007 codes
    rename_dict_00 = {v: k for k, v in cfg["geo_columns_00"].items()}
    screening_dataset = screening_dataset.rename(columns=rename_dict_00)
    screening_dataset = screening_dataset[
        list(rename_dict.values()) + list(rename_dict_00.values())
    ].copy()

    # We want one per household, not per interviewee
    screening_dataset["province_code"] = screening_dataset.province_code.astype(int)
    screening_dataset["kabupaten_code"] = screening_dataset.kabupaten_code.astype(int)
    screening_dataset["kecamatan_code"] = screening_dataset.kecamatan_code.astype(int)
    # Generate combined code for easy matching with GAMD boundary data.
    # The code is province (2 digits) + kabupaten (2 digits) + kecamatan (3 digits)
    screening_dataset["gadm_fullcode_07"] = (
        screening_dataset.province_code.astype(str).str.zfill(2)
        + screening_dataset.kabupaten_code.astype(str).str.zfill(2)
        + screening_dataset.kecamatan_code.astype(str).str.zfill(3)
    ).astype(int)

    screening_dataset["gadm_fullcode_00"] = (
        screening_dataset.province_code_00.astype(int).astype(str).str.zfill(2)
        + screening_dataset.kabupaten_code_00.astype(int).astype(str).str.zfill(2)
        + screening_dataset.kecamatan_code_00.astype(int).astype(str).str.zfill(3)
    ).astype(int)

    mapping_00 = generate_mapping_data(original_col="kecid00")
    mapping_07 = generate_mapping_data(original_col="kecid07")

    screening_dataset["wave"] = "IFLS4"

    screening_dataset = screening_dataset.merge(
        mapping_07,
        left_on=["gadm_fullcode_07"],
        right_on=["kecid07"],
        how="left",
        validate="many_to_one",
    )
    screening_dataset["gadm_fullcode_07_2014"] = screening_dataset.gadm_fullcode_2014
    screening_dataset.drop(columns=["kecid07", "gadm_fullcode_2014"], inplace=True)

    screening_dataset = screening_dataset.merge(
        mapping_00,
        left_on=["gadm_fullcode_00"],
        right_on=["kecid00"],
        how="left",
        validate="many_to_one",
    )
    screening_dataset["gadm_fullcode"] = screening_dataset.gadm_fullcode_07_2014.fillna(
        screening_dataset.gadm_fullcode_2014
    )
    # Multiple map indicates that a single 2007 district was mapping to many 2014 ones; detect by checking for comma
    screening_dataset["multiple_kec_remap"] = screening_dataset.gadm_fullcode.apply(
        lambda x: 1 if isinstance(x, str) and "," in x else 0
    )
    screening_dataset = screening_dataset[
        [
            "gadm_fullcode",
            "wave",
            "hhid",
            "province_code",
            "kabupaten_code",
            "kecamatan_code",
            "multiple_kec_remap",
        ]
    ].copy()
    return screening_dataset


DATE_COLUMNS = [
    "pidlink",
    "hhid",
    "wave",
    "interview_datetime",
    "day",
    "month",
    "year",
    "hour_start",
    "hour_end",
]


def parse_ifls4_survey_info() -> pd.DataFrame:
    path = Path(RAW_IFLS_EXTRACTED) / "IFLS4" / "hh07" / "b3b_cov.dta"
    survey_info = read_stata_df(path, convert_categoricals=False)
    mapping = {
        "year": "ivwyr",
        "month": "ivwmth",
        "day": "ivwday",
        "hour_start": "beghr",
        "hour_end": "endhr",
    }
    # Each person might have been visited more than once; read the latest date based
    # based on the visit times so we have right survey time for temperature matching
    for col, prefix in mapping.items():
        # Pick the right columb based on the numvis variable
        survey_info[col] = np.select(
            [
                survey_info.numvis == 1,
                survey_info.numvis == 2,
                survey_info.numvis == 3,
            ],
            [
                survey_info[f"{prefix}1"],
                survey_info[f"{prefix}2"],
                survey_info[f"{prefix}3"],
            ],
        )
    # The raw data just has '8' or '7' for the year
    survey_info["year"] = survey_info.year + 2000

    survey_info["hhid"] = survey_info["hhid07"]
    survey_info["wave"] = "IFLS4"
    # Print hh_id if interview_datetime is null
    # For IFLS4 four households have incorrect survey dates:
    # 003134102 -> Feb 30
    # 073190006 ->  Feb 31
    # 106150010 ->  Nov 31
    # 231300004 ->  Nov 31
    # OLD Fix the above to make it the last day of the month before datetime generation
    # survey_info.loc[survey_info.pidlink == "003134102", "day"] = 28
    # survey_info.loc[survey_info.pidlink == "073190006", "day"] = 28
    # survey_info.loc[survey_info.pidlink == "106150010", "day"] = 30
    # survey_info.loc[survey_info.pidlink == "231300004", "day"] = 30

    # The above was for time variable from a different  dataset; since updating,
    # only the row below has an issue with a feb 30 date
    survey_info.loc[survey_info.pidlink == "275110002", "day"] = 28

    survey_info["interview_datetime"] = pd.to_datetime(
        dict(
            year=survey_info.year,
            month=survey_info.month,
            day=survey_info.day,
            hour=survey_info.hour_end,
        ),
        errors="coerce",
    )
    # if interview_datetime is null, log the undderlying variables using error level
    null_datetime = survey_info[survey_info["interview_datetime"].isna()]
    for observation in null_datetime.itertuples():
        log(
            f"Null interview datetime for pidlink {observation.pidlink} with year {observation.year}, month {observation.month}, day {observation.day}, hour_end {observation.hour_end}",
            "ERROR",
        )

    null_interview_pids = survey_info[survey_info["interview_datetime"].isna()][
        "pidlink"
    ].tolist()
    log(
        f"HHIDs with null interview datetime: {null_interview_pids}",
        "WARNING" if null_interview_pids else "DEBUG",
    )
    return survey_info[DATE_COLUMNS]


def parse_ifls5_survey_info() -> pd.DataFrame:
    path = Path(RAW_IFLS_EXTRACTED) / "IFLS5" / "hh14" / "b3b_time.dta"
    survey_info = read_stata_df(path, convert_categoricals=False)
    survey_info["year"] = survey_info.ivwyr
    survey_info["month"] = survey_info.ivwmth
    survey_info["day"] = survey_info.ivwday
    survey_info["hour_start"] = survey_info.beghr
    survey_info["hour_end"] = survey_info.endhr

    survey_info["interview_datetime"] = pd.to_datetime(
        dict(
            year=survey_info.year,
            month=survey_info.month,
            day=survey_info.day,
            hour=survey_info.hour_end,
        ),
        errors="coerce",
    )
    # if interview_datetime is null, log the undderlying variables using error level
    null_datetime = survey_info[survey_info["interview_datetime"].isna()]
    for observation in null_datetime.itertuples():
        log(
            f"Null interview datetime for pidlink {observation.pidlink} with year {observation.year}, month {observation.month}, day {observation.day}, hour_end {observation.hour_end}",
            "ERROR",
        )

    survey_info["hhid"] = survey_info["hhid14"]
    survey_info["wave"] = "IFLS5"
    # TODO: use latest time for robustness as well
    # Sort by pidlink and interview time, then drop duplicates to keep the earliest interview for each person
    survey_info = survey_info.dropna(subset=["interview_datetime"], how="all")
    survey_info = survey_info.sort_values(by=["pidlink", "interview_datetime"])
    survey_info = survey_info.drop_duplicates(subset=["pidlink"], keep="first")
    return survey_info[DATE_COLUMNS]


def main() -> None:
    survey_datetime_ifls4 = parse_ifls4_survey_info()
    survey_datetime_ifls5 = parse_ifls5_survey_info()
    survey_both = pd.concat(
        [survey_datetime_ifls4, survey_datetime_ifls5], ignore_index=True
    )
    # Catch missing hours tagged as 99
    survey_both.loc[survey_both.hour_start == 99, "hour_start"] = 0
    survey_both.loc[survey_both.hour_end == 99, "hour_end"] = 0

    geo_ifls4 = parse_geo_codes_ifls4()
    geo_ifls5 = parse_geo_codes_ifls5()

    geo_both = pd.concat([geo_ifls4, geo_ifls5], ignore_index=True)
    geo_both = geo_both.drop_duplicates(subset=["hhid", "wave"], keep="first")

    out = survey_both.merge(
        geo_both, on=["hhid", "wave"], how="left", validate="many_to_one"
    )
    # Validate against schema
    out = INDIVIDUALS_SCHEMA.validate(out)
    out.to_parquet(GENERATED_DATA / "01_individuals.parquet", index=False)
    log(
        f"wrote {len(out):,} individual-wave rows to {GENERATED_DATA / '01_individuals.parquet'}"
    )


if __name__ == "__main__":
    main()
