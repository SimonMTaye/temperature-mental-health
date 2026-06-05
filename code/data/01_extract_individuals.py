"""
Extract individual-level IFLS4 + IFLS5 panel data with geographic identifiers

The data serves as a panel to which we will merge other variables and use to fetch
temperature data

Output: data/generated/01_individuals.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from decimal import Decimal, InvalidOperation

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


def clean_gadm_code(value: object) -> str:
    """Return canonical integer-code strings, preserving comma-separated mappings."""
    if pd.isna(value):
        raise ValueError("gadm_fullcode cannot be missing")

    def clean_part(part: str) -> str:
        part = part.strip()
        try:
            numeric = Decimal(part)
        except InvalidOperation:
            raise ValueError(f"invalid GADM code component: {part}") from None
        if numeric != numeric.to_integral_value():
            raise ValueError(f"non-integer GADM code component: {part}")
        return str(int(numeric))

    return ",".join(clean_part(part) for part in str(value).split(","))


def parse_geo_codes_ifls5() -> pd.DataFrame:
    """
    Extract household geography from a wave's screening file.
    Rename and standardize
    """
    cfg = IFLS5
    screening_dataset = read_stata_df(
        cfg["file"],  # ty:ignore[invalid-argument-type]
        convert_categoricals=False,
    )
    # Reverse the dict so mapping works
    rename_dict = {v: k for k, v in cfg["geo_columns"].items()}  # ty:ignore[unresolved-attribute]
    screening_dataset = screening_dataset.rename(columns=rename_dict)
    screening_dataset = screening_dataset[cfg["geo_columns"].keys()].copy()  # ty:ignore[unresolved-attribute]

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
    ).map(clean_gadm_code)
    screening_dataset["wave"] = "IFLS5"
    # Below is an indicator column to flag remapping complication when converting IFLS4 Admin codes to 5
    # Not relevant for IFLS5 so set to 0
    screening_dataset["kecamatan_code_map"] = "not_applicable"
    return screening_dataset[
        [
            "hhid",
            "province_code",
            "kabupaten_code",
            "gadm_fullcode",
            "kecamatan_code",
            "wave",
            "kecamatan_code_map",
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


def unique_bps_mapping() -> pd.DataFrame:
    mapping = read_stata_df(
        Path(RAW_IFLS_EXTRACTED)
        / "IFLS5"
        / "IFLS5_all_doc"
        / "IFLS5_BPS_2014_codes"
        / "kec_9899000714.dta"
    )
    mapping = mapping[["kecid07", "kecid14"]].copy()
    # Count the number of times kecio07 appears; do the same for kecid14
    mapping["count_07"] = mapping.groupby("kecid07")["kecid07"].transform("count")
    mapping["count_14"] = mapping.groupby("kecid14")["kecid14"].transform("count")
    # Filter to only rows where both counts are 1, meaning it's a unique mapping
    mapping = mapping[(mapping["count_07"] == 1) & (mapping["count_14"] == 1)]
    mapping = mapping[["kecid07", "kecid14"]].copy()
    mapping["kecid07"] = mapping.kecid07.astype(int)
    mapping["kecid14"] = mapping.kecid14.astype(int)
    return mapping


def parse_geo_codes_ifls4() -> pd.DataFrame:
    """
    Extract household geography from a wave's screening file.
    Rename and standardize

    Does additional work to convert IFLS4 2007 BPS codes to 2014 BPS codes compatible with GADM boundaries

    """
    cfg = IFLS4
    screening_dataset = read_stata_df(
        cfg["file"],  # ty:ignore[invalid-argument-type]
        convert_categoricals=False,
    )
    # Reverse the dict so mapping works
    rename_dict = {v: k for k, v in cfg["geo_columns"].items()}  # ty:ignore[unresolved-attribute]
    screening_dataset = screening_dataset.rename(columns=rename_dict)
    # Rename 2000 codes as well since mapping dict doesn't have all the 2007 codes
    rename_dict_00 = {v: k for k, v in cfg["geo_columns_00"].items()}  # ty:ignore[unresolved-attribute]
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

    screening_dataset = screening_dataset.merge(
        unique_bps_mapping(),
        left_on=["gadm_fullcode_07"],
        right_on=["kecid07"],
        how="left",
        validate="many_to_one",
        indicator="bps_mapping",
    )

    screening_dataset["bps_mapping"] = screening_dataset.bps_mapping.map(
        {"both": "unique", "left_only": "no_2014_match", "right_only": "no_2007_match"}
    )
    # For the good matches, no further work is needed
    good_matches = screening_dataset[screening_dataset.bps_mapping == "unique"]
    good_matches["gadm_fullcode"] = good_matches.kecid14

    bad_matches = screening_dataset[screening_dataset.bps_mapping != "unique"]
    bad_matches = bad_matches.drop(columns=["kecid07", "kecid14"])
    log(
        f"{len(good_matches)} rows with unique BPS mapping, {len(bad_matches)} rows with non-unique or missing BPS mapping",
        "INFO",
    )

    # For the bad matches, let us first see if they moved and if they didn't, we can find them based on their IFLS5 code tos ave us the work
    household_mapping = read_stata_df(
        Path(RAW_IFLS_EXTRACTED) / "IFLS5" / "hh14" / "htrack.dta",
        convert_categoricals=False,
    )

    # Standardize the column names for geo codes (exclude hhid to keep hhid14 intact)
    # We use IFLS5 geo codes since we are using IFLS5 household tracking
    geo_rename = {
        v: k
        for k, v in IFLS5["geo_columns"].items()  # ty:ignore[unresolved-attribute]
        if k != "hhid"
    }
    household_mapping = household_mapping.rename(columns=geo_rename)
    household_mapping = household_mapping.dropna(
        subset=["province_code", "kabupaten_code", "kecamatan_code"]
    ).copy()
    household_mapping["kecid14"] = (
        household_mapping.province_code.astype(int).astype(str).str.zfill(2)
        + household_mapping.kabupaten_code.astype(int).astype(str).str.zfill(2)
        + household_mapping.kecamatan_code.astype(int).astype(str).str.zfill(3)
    ).astype(int)
    household_mapping = household_mapping[
        ["hhid07", "hhid14", "kecid14", "mover14"]
    ].copy()
    # Keep only those who haven't moved or moved within a kecamatan
    household_mapping = household_mapping[household_mapping.mover14 <= 2].copy()
    # Keep only households that existed in 2007 IFLS4
    household_mapping = household_mapping[household_mapping.hhid07.notna()].copy()
    # Drop duplicate hhid07 (household splits) to ensure 1:1 merge
    # Note: we have already restricted to non-movers so it is immaterial which household we draw the mapping
    # code from
    household_mapping = household_mapping.drop_duplicates(
        subset=["hhid07"], keep="first"
    )

    bad_matches = bad_matches.merge(
        household_mapping,
        left_on="hhid",
        right_on="hhid07",
        how="left",
        validate="1:1",
        indicator="household_mapping",
    )
    # Drop new households, we don't care about them
    bad_matches = bad_matches[bad_matches.household_mapping != "right_only"].copy()
    bad_matches["household_mapping"] = bad_matches.household_mapping.map(
        {
            "both": "mapped",
            "left_only": "no_ifls5_match",
        }
    )
    old_household_matches = bad_matches[
        bad_matches.household_mapping == "mapped"
    ].copy()
    old_household_matches["gadm_fullcode"] = old_household_matches.kecid14
    old_household_matches["bps_mapping"] = "unique"
    good_matches = pd.concat([good_matches, old_household_matches], ignore_index=True)

    # Let us do a "voting" approach where we derive a mappign based on our data
    # Basically, for each kecid07, we pick the most common kecid14 amongst the good matches
    # And use that to assign a good match

    bad_matches = bad_matches[bad_matches.household_mapping == "no_ifls5_match"].copy()
    log(
        f"{len(old_household_matches)} bad matches resolved by household tracking, {len(bad_matches)} bad matches with no IFLS5 household match",
        "INFO",
    )

    # Final matching strategy is to use 2014 codes derived from our good matches. Here, we use voting
    # to determine best match
    good_match_mapping = (
        good_matches.groupby("gadm_fullcode_07")["gadm_fullcode"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .assign(vote_kecid14=lambda df: df.gadm_fullcode)
        .filter(items=["gadm_fullcode_07", "vote_kecid14"])
        .copy()
    )
    assert good_match_mapping.vote_kecid14.is_unique, (
        "vote_kecid14 should be unique in good match mapping"
    )
    assert good_match_mapping.vote_kecid14.isna().sum() == 0, (
        "gadm_fullcode should not have nulls in good match mapping"
    )

    bad_matches = bad_matches.merge(
        good_match_mapping,
        on="gadm_fullcode_07",
        how="left",
        validate="m:1",
        indicator="good_match_mapping",
    )
    bad_matches["good_match_mapping"] = bad_matches.good_match_mapping.map(
        {
            "both": "good_match_found",
            "left_only": "no_good_match",
            "right_only": "no_2007_match",
        }
    )
    bad_matches = bad_matches[bad_matches.good_match_mapping != "no_2007_match"].copy()

    success_count = (bad_matches.good_match_mapping == "good_match_found").sum()
    failure_count = (bad_matches.good_match_mapping == "no_good_match").sum()
    log(
        f"{success_count} bad matches resolved by voting for most common mappign, {failure_count} bad matches with no good match mapping",
    )
    vote_matches = bad_matches[
        bad_matches.good_match_mapping == "good_match_found"
    ].copy()
    vote_matches["gadm_fullcode"] = vote_matches.vote_kecid14
    vote_matches["bps_mapping"] = "voting_based"
    good_matches = pd.concat([good_matches, vote_matches], ignore_index=True)

    bad_matches = bad_matches[bad_matches.good_match_mapping == "no_good_match"].copy()

    log(
        f"{len(bad_matches)} bad matches remain after household tracking and voting. First kecamatan code will be selected arbitrarily for these households",
        "INFO",
    )

    bad_matches["gadm_fullcode_00"] = (
        bad_matches.province_code_00.astype(int).astype(str).str.zfill(2)
        + bad_matches.kabupaten_code_00.astype(int).astype(str).str.zfill(2)
        + bad_matches.kecamatan_code_00.astype(int).astype(str).str.zfill(3)
    ).astype(int)

    mapping_00 = generate_mapping_data(original_col="kecid00")
    mapping_07 = generate_mapping_data(original_col="kecid07")

    bad_matches["wave"] = "IFLS4"

    bad_matches = bad_matches.merge(
        mapping_07,
        left_on=["gadm_fullcode_07"],
        right_on=["kecid07"],
        how="left",
        validate="many_to_one",
    )
    # gadm_fullcode_2014 is the kecamaten code for 2014; it may be a comma separated value if one 2007 kecamatan is matched with many 2014 ones
    bad_matches["gadm_fullcode_07_2014"] = bad_matches.gadm_fullcode_2014
    bad_matches.drop(columns=["kecid07", "gadm_fullcode_2014"], inplace=True)

    bad_matches = bad_matches.merge(
        mapping_00,
        left_on=["gadm_fullcode_00"],
        right_on=["kecid00"],
        how="left",
        validate="many_to_one",
    )
    bad_matches["gadm_fullcode"] = bad_matches.gadm_fullcode_07_2014.fillna(
        bad_matches.gadm_fullcode_2014
    )
    # Randomly pick the first code in cases of multiple mapping; we will flag these cases in the end for sensitivity analysis
    bad_matches["gadm_fullcode"] = bad_matches.gadm_fullcode.apply(
        lambda x: int(x.split(",")[0]) if isinstance(x, str) else x
    )
    bad_matches["bps_mapping"] = "select_first"
    # Multiple map indicates that a single 2007 district was mapping to many 2014 ones; detect by checking for comma

    screening_dataset = pd.concat([good_matches, bad_matches], ignore_index=True)
    screening_dataset["kecamatan_code_map"] = screening_dataset.bps_mapping

    screening_dataset = screening_dataset[
        [
            "gadm_fullcode",
            "wave",
            "hhid",
            "province_code",
            "kabupaten_code",
            "kecamatan_code",
            "kecamatan_code_map",
        ]
    ].copy()
    screening_dataset["wave"] = "IFLS4"
    screening_dataset["gadm_fullcode"] = screening_dataset.gadm_fullcode.map(
        clean_gadm_code
    )
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
