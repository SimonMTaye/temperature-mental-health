"""Build per-(pidlink, wave) stressor / covariate variables for IFLS4 and IFLS5.

Outputs: data/generated/stressors.parquet
  pidlink, wave, age, sex, married, widowed, edu_yrs, urban,
  hh_size, pce_log, pce_quintile,
  loan_rejected,
  disaster_5yr, disaster_severe_5yr,
  hh_with_adl_limit (proxy: caregiver burden)

Notes
-----
- "PCE" for IFLS4 = pre-computed nominal monthly per-capita expenditure (pce07nom.dta).
- "PCE" for IFLS5 = computed here from b1_ks0/ks2/ks3 (food + non-food monthly).
- Disaster variables are HH-level (b2_nd1) — broadcast to all adults in HH.
- Quintiles computed within wave so they're comparable across waves.
"""

import numpy as np
import pandas as pd

# Make _sentinels.py importable when this script is run directly
from data.config import GENERATED_DATA, IFLS4_FOLDER, IFLS5_FOLDER, RAW_IFLS_EXTRACTED  # noqa: E402
from data._sentinels import clean_age, clean_categorical  # noqa: E402
from data._schemas import STRESSORS_SCHEMA  # noqa: E402
from data._stata import read_stata_df  # noqa: E402
from library.log import log  # noqa: E402

IFLS_FOLDERS = {
    "IFLS4": IFLS4_FOLDER,
    "IFLS5": IFLS5_FOLDER,
}

HHID_COLUMNS = {
    "IFLS4": "hhid07",
    "IFLS5": "hhid14",
}

MISSING_CATEGORY = -99

# Distinct sentinel for education-year values whose AR16 level or AR17 grade is
# unknown. This is separate from MISSING_CATEGORY because edu_yrs is a modeled
# continuous control, while religion/ethnicity are categorical codes.
MISSING_EDU_YEARS = -100

# AR16 identifies the respondent's education level. These values are the
# maximum years credited when AR17 says the respondent graduated from that level
# (AR17 == 7). The levels follow the IFLS AR16 labels and Indonesian schooling
# structure: SD/MI/Paket A = 6, SMP/MTs/Paket B = 9, SMA/MA/SMK/Paket C = 12,
# S1/open university = 16, S2 = 18, and S3 = 21. Keep D1/D2/D3 at the current
# project value of 16 because AR16 pools those diplomas.
EDU_LEVEL_YEARS = {
    1: 0,  # Unschooled
    90: 0,  # Kindergarten
    2: 6,  # Grade school / SD
    11: 6,  # Education A / Paket A, SD equivalent
    72: 6,  # Madrasah Ibtidaiyah, SD equivalent
    3: 9,  # General junior high / SMP
    4: 9,  # Vocational junior high
    12: 9,  # Education B / Paket B, SMP equivalent
    73: 9,  # Madrasah Tsanawiyah, SMP equivalent
    5: 12,  # General senior high / SMA
    6: 12,  # Vocational senior high / SMK
    15: 12,  # Education C / Paket C, SMA equivalent
    74: 12,  # Madrasah Aliyah, SMA equivalent
    60: 16,  # Diploma D1/D2/D3 are pooled; retain the project max-year value.
    61: 16,  # University S1
    62: 18,  # University S2
    63: 21,  # University S3
    13: 16,  # Open University, treated as S1-equivalent
    14: 16,  # Moslem School (Pesantren), treated as S1-equivalent
    17: 16,  # School for the disabled, treated as S1-equivalent
    95: 16,  # Other, specify; treated as S1-equivalent
}

# For non-graduated AR17 values, use the largest observed AR17 grade below 7 for
# each AR16 level as the within-level maximum. These constants were computed
# from pooled IFLS4 and IFLS5 household roster files:
# data/raw/IFLS/extracted/IFLS4/hh07/bk_ar1.dta and
# data/raw/IFLS/extracted/IFLS5/hh14/bk_ar1.dta. AR17 == 7 is "graduated" and
# AR17 98/99 are missing codes, so neither contributes to this dictionary.
EDU_LEVEL_MAX_GRADE = {
    2: 6,
    3: 3,
    4: 3,
    5: 3,
    6: 3,
    11: 1,
    12: 3,
    13: 5,
    14: 6,
    15: 3,
    17: 5,
    60: 5,
    61: 6,
    62: 3,
    63: 3,
    72: 6,
    73: 3,
    74: 4,
    90: 6,
    95: 4,
}

##AR15               Religion of HH member
#
#                   missing                                      3,936      5.39
#                   1. Islam                                    61,341     84.01
#                   2. Protestant                                3,200      4.38
#                   3. Catholic                                  1,196      1.64
#                   4. Hinduism                                  2,959      4.05
#                   5. Buddhism                                    289      0.40
#                   7. Konghucu                                      8      0.01
#                   95. Other, specify                              67      0.09
#                   99. missing, specify                           20      0.09

##AR15D              Ethnic Group
##
#                   missing                                      3,936      5.39
#                   1. Javanese                                 27,895     38.20
#                   2. Sundanese                                 8,637     11.83
#                   3. Balinese                                  2,866      3.93
#                   4. Batak                                     3,155      4.32
#                   5. Bugis                                     2,784      3.81
#                   6. Chinese                                     645      0.88
#                   7. Maduranese                                1,987      2.72
#                   8. Sasak                                     2,736      3.75
#                   9. Minang                                    3,169      4.34
#                   10. Banjar                                   2,372      3.25
#                   11. Bima-Dompu                               1,307      1.79
#                   12. Makassar                                 1,016      1.39
#                   13. Nias                                       284      0.39
#                   14. Palembang                                  379      0.52
#                   15. Sumbawa                                    310      0.42
#                   16. Toraja                                     566      0.78
#                   17. Betawi                                   2,537      3.47
#                   18. Dayak                                       62      0.08
#                   19. Melayu                                     643      0.88
#                   20. Komering                                    92      0.13
#                   21. Ambon                                       32      0.04
#                   22. Manado                                      24      0.03
#                   23. Aceh                                        76      0.10
#                   25. SumbagSel lain                           2,688      3.68
#                   26. Banten                                     236      0.32
#                   27. Cirebon                                  1,480      2.03
#                   28. Gorontalo                                    3      0.00
#                   95. Other, specify                           1,046      1.43
#                   99. Missing                                     53      0.07
#                   total                                       73,016    100.00
#

# AR16               HHM highest level of education
#
#                   missing                                      3,936      5.39
#                   1. Unschooled                                9,825     13.46
#                   2. Grade school                             23,368     32.00
#                   3. General jr. high                          8,984     12.30
#                   4. Vocational jr. high                         543      0.74
#                   5. General sr. high (SLA)                    8,268     11.32
#                   6. Vocational sr. high (SMK)                 5,764      7.89
#                   11. Education A                                 28      0.04
#                   12. Education B                                 70      0.10
#                   13. Open University                             26      0.04
#                   14. Moslem School (Pesantren)                  256      0.35
#                   15. Education C                                 82      0.11
#                   17. School for the disabled                     26      0.04
#                   60. Diploma (D1,D2,D3)                       2,024      2.77
#                   61. University S1                            3,782      5.18
#                   62. University S2                              189      0.26
#                   63. University S3                               10      0.01
#                   72. Madrasah Ibtidaiyah                      1,193      1.63
#                   73. Madrasah Tsanawiyah                      2,074      2.84
#                   74. Madrasah Aliyah                          1,166      1.60
#                   90. Kindergarten                               517      0.71
#                   95. Other, specify                              10      0.01
#                   98. Don't know                                 857      1.17
#                   99. Missing                                     18      0.02
#                   total                                       73,016    100.00
#


# AR13               Marital status
#
#                   missing                                      3,936      5.39
#                   1. Unmarried                                28,751     39.38
#                   2. Married                                  35,539     48.67
#                   3. Separated, estranged                        305      0.42
#                   4. Divorced                                  1,176      1.61
#                   5. Widow, widower                            3,179      4.35
#                   8. Don't know                                  116      0.16
#                   9. Missing                                      14      0.02
#                   total                                       73,016    100.00
def _education_years(edu_lvl: pd.Series, edu_grade: pd.Series) -> pd.Series:
    """Approximate schooling years from IFLS AR16 level and AR17 grade."""
    lvl = pd.to_numeric(edu_lvl, errors="coerce")
    grade = pd.to_numeric(edu_grade, errors="coerce")
    max_years = lvl.map(EDU_LEVEL_YEARS)
    max_grade = lvl.map(EDU_LEVEL_MAX_GRADE)

    # Start with all empty years
    years = pd.Series(MISSING_EDU_YEARS, index=edu_lvl.index, dtype="Int64")
    # Verify that the level can be interpreted (i.e. it is not other or missing)
    valid_level = max_years.notna()

    # If a non-missing level is there but the "grade" is a incomplete / not in school (i.e. 96) set to 0 years
    years.loc[valid_level & grade.eq(96)] = 0

    # If grade == 7 that means, the level of schooling is completed
    graduated = valid_level & grade.eq(7)
    # Set years to that level's max years for those who graduated (eg. 12 for high school)
    years.loc[graduated] = max_years.loc[graduated].astype("Int64")

    # If not graduated, then mark as in progress (i.e. they are still working to completed the level of educated schooling)
    in_progress = valid_level & grade.between(0, 6) & max_grade.notna()
    in_progress_years = (
        max_years.loc[in_progress]
        - (max_grade.loc[in_progress] - grade.loc[in_progress])
    ).clip(lower=0)
    years.loc[in_progress] = in_progress_years.astype("Int64")

    return years


def _demographics_from_roster(ar: pd.DataFrame, *, hhid_col_name: str) -> pd.DataFrame:
    """Normalize IFLS roster demographics to person-wave covariates."""
    cols_keep = [
        c
        for c in [
            "pidlink",
            hhid_col_name,
            "ar07",
            "ar09",
            "ar13",
            "ar15",
            "ar15d",
            "ar16",
            "ar17",
        ]
        if c in ar.columns
    ]
    ar = ar[cols_keep].rename(
        columns={
            "ar07": "sex_raw",
            "ar09": "age",
            "ar13": "marital_raw",
            "ar15": "religion_raw",
            "ar15d": "ethnicity_raw",
            "ar16": "edu_lvl",
            "ar17": "edu_grade",
            hhid_col_name: "hhid",
        }
    )
    ar = ar.dropna(subset=["pidlink"]).drop_duplicates("pidlink")
    ar["age"] = clean_age(ar.age)
    if "marital_raw" in ar.columns:
        ar["marital_raw"] = clean_categorical(ar.marital_raw, digits=1)
    ar["sex"] = (ar.sex_raw == 3).map({True: "F", False: "M"})  # 1=M,3=F per IFLS docs
    ar["female"] = (ar.sex_raw == 3).astype(int)
    ar["female_missing"] = (ar.sex_raw.isna()).astype(int)
    ar["married"] = (
        (ar.get("marital_raw").eq(2)).astype(int) if "marital_raw" in ar.columns else 0
    )
    ar["widowed"] = (
        (ar.get("marital_raw").eq(5)).astype(int) if "marital_raw" in ar.columns else 0
    )
    ar["divorced"] = (
        (ar.get("marital_raw").isin([3, 4])).astype(int)
        if "marital_raw" in ar.columns
        else 0
    )
    ar["edu_yrs"] = _education_years(ar.edu_lvl, ar.edu_grade)
    ar["edu_yrs_missing"] = (ar.edu_yrs.eq(MISSING_EDU_YEARS)).astype(int)
    ar["religion"] = (
        clean_categorical(ar.religion_raw, digits=2)
        .fillna(MISSING_CATEGORY)
        .astype("Int64")
    )
    ar["ethnicity"] = (
        clean_categorical(ar.ethnicity_raw, digits=2)
        .fillna(MISSING_CATEGORY)
        .astype("Int64")
    )
    ar["religion_missing"] = ar.religion.eq(MISSING_CATEGORY).astype(int)
    ar["ethnicity_missing"] = ar.ethnicity.eq(MISSING_CATEGORY).astype(int)

    return ar[
        [
            "pidlink",
            "hhid",
            "age",
            "sex",
            "female",
            "female_missing",
            "married",
            "widowed",
            "divorced",
            "edu_yrs",
            "edu_yrs_missing",
            "religion",
            "religion_missing",
            "ethnicity",
            "ethnicity_missing",
        ]
    ]


def _ifls5_demographics() -> pd.DataFrame:
    ar = read_stata_df(
        IFLS5_FOLDER / "bk_ar1.dta",
        convert_categoricals=False,
    )
    return _demographics_from_roster(ar, hhid_col_name=HHID_COLUMNS["IFLS5"])


def _ifls4_demographics() -> pd.DataFrame:
    ar = read_stata_df(
        IFLS4_FOLDER / "bk_ar1.dta",
        convert_categoricals=False,
    )
    return _demographics_from_roster(ar, hhid_col_name=HHID_COLUMNS["IFLS4"])


def _ifls5_pce() -> pd.DataFrame:
    """Approx monthly per-capita expenditure for IFLS5.
    weekly -> *4.33; monthly stays as is. Sums major modules.
    Quintile within IFLS5 sample.
    """
    # b1_ks0 has weekly food (ks02a) + monthly non-food various items at HH level
    ks0 = read_stata_df(IFLS5_FOLDER / "b1_ks0.dta", convert_categoricals=False)
    # Aggregate any monetary monthly columns we can identify
    money_cols = [
        c for c in ks0.columns if c.startswith("ks") and ks0[c].dtype.kind in "if"
    ]
    # ks02a/ks04b are weekly food; rest monthly
    weekly_food = ks0.get("ks02a", 0).fillna(0) * 4.33
    monthly_other = pd.Series(0.0, index=ks0.index)
    for c in money_cols:
        if c in ("ks02a", "ks04b"):
            continue
        monthly_other = monthly_other + ks0[c].fillna(0)
    ks0["x_total_mo"] = weekly_food + monthly_other

    # household size: count of pidlink in bk_ar1
    ar = read_stata_df(IFLS5_FOLDER / "bk_ar1.dta", convert_categoricals=False)
    hhsize = ar.groupby("hhid14", as_index=False).agg(hhsize=("pidlink", "size"))
    out = ks0[["hhid14", "x_total_mo"]].merge(hhsize, on="hhid14", validate="m:1")
    out["pce"] = out.x_total_mo / out.hhsize.replace(0, np.nan)
    out["wave"] = "IFLS5"
    return out[["hhid14", "hhsize", "pce", "wave"]].rename(columns={"hhid14": "hhid"})


def _ifls4_pce() -> pd.DataFrame:
    df = read_stata_df(
        RAW_IFLS_EXTRACTED / "consumption/pce/pce-1993-1997_2000-2007/pce07nom.dta",
        convert_categoricals=False,
    )
    df = df[["hhid07", "hhsize", "pce"]].rename(columns={"hhid07": "hhid"})
    df["wave"] = "IFLS4"
    return df


def _disaster_from_df(
    nd: pd.DataFrame, *, hhid_col_name: str, wave: str
) -> pd.DataFrame:
    nd["disaster_5yr"] = (nd.nd01 == 1).astype(int)
    nd["disaster_severe_5yr"] = (nd.nd02 == 1).astype(int)
    nd = nd[[hhid_col_name, "disaster_5yr", "disaster_severe_5yr"]].rename(
        columns={hhid_col_name: "hhid"}
    )
    nd = nd.drop_duplicates("hhid")
    nd["wave"] = wave
    return nd


def _disaster(wave: str) -> pd.DataFrame:
    nd = read_stata_df(IFLS_FOLDERS[wave] / "b2_nd1.dta", convert_categoricals=False)
    return _disaster_from_df(nd, hhid_col_name=HHID_COLUMNS[wave], wave=wave)


def _loan_rejected_from_df(
    bh: pd.DataFrame, *, hhid_col_name: str, wave: str
) -> pd.DataFrame:
    bh["loan_rejected"] = (bh.bh04 == 1).astype(int)
    bh = (
        bh[[hhid_col_name, "loan_rejected"]]
        .rename(columns={hhid_col_name: "hhid"})
        .drop_duplicates("hhid")
    )
    bh["wave"] = wave
    return bh


def _loan_rejected(wave: str) -> pd.DataFrame:
    bh = read_stata_df(IFLS_FOLDERS[wave] / "b2_bh.dta", convert_categoricals=False)
    return _loan_rejected_from_df(bh, hhid_col_name=HHID_COLUMNS[wave], wave=wave)


def _add_pce_fields(pce: pd.DataFrame) -> pd.DataFrame:
    return pce.assign(
        pce=lambda df: df.pce.replace([0, np.inf, -np.inf], np.nan),
        pce_log=lambda df: np.log(df.pce),
        pce_quintile=lambda df: df.groupby("wave")["pce"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1
        ),
    )


def _finalize_stressors(out: pd.DataFrame) -> pd.DataFrame:
    return out.assign(
        disaster_5yr=lambda df: df.disaster_5yr.fillna(0).astype(int),
        disaster_severe_5yr=lambda df: df.disaster_severe_5yr.fillna(0).astype(int),
        loan_rejected=lambda df: df.loan_rejected.fillna(0).astype(int),
    ).pipe(STRESSORS_SCHEMA.validate)


def main() -> None:
    # Demographics
    demo = pd.concat(
        [
            _ifls4_demographics().assign(wave="IFLS4"),
            _ifls5_demographics().assign(wave="IFLS5"),
        ],
        ignore_index=True,
    )
    log(f"demographics: {len(demo):,}")

    # PCE
    pce = pd.concat([_ifls4_pce(), _ifls5_pce()], ignore_index=True).pipe(
        _add_pce_fields
    )
    log(f"pce: {len(pce):,}")

    # HH-level shocks
    disaster = pd.concat([_disaster("IFLS4"), _disaster("IFLS5")], ignore_index=True)
    loan = pd.concat(
        [_loan_rejected("IFLS4"), _loan_rejected("IFLS5")], ignore_index=True
    )
    log(f"disaster: {len(disaster):,}, loan: {len(loan):,}")

    # Merge by (pidlink, wave) — most are HH-level so first attach to demo via hhid
    out = (
        demo.merge(
            pce[["hhid", "wave", "hhsize", "pce", "pce_log", "pce_quintile"]],
            on=["hhid", "wave"],
            how="left",
            validate="m:1",
        )
        .merge(disaster, on=["hhid", "wave"], how="left", validate="m:1")
        .merge(loan, on=["hhid", "wave"], how="left", validate="m:1")
        .pipe(_finalize_stressors)
    )
    out.to_parquet(GENERATED_DATA / "22_stressors.parquet", index=False)
    log(
        f"\nwrote {len(out):,} stressor rows to {GENERATED_DATA / '22_stressors.parquet'}"
    )
    log("stressor prevalence by wave:", "DEBUG")
    log(
        out.groupby("wave")
        .agg(
            n=("pidlink", "size"),
            age_med=("age", "median"),
            female_pct=("sex", lambda s: 100 * (s == "F").mean()),
            edu_yrs_med=("edu_yrs", "median"),
            widowed_pct=("widowed", lambda s: 100 * s.mean()),
            pce_log_med=("pce_log", "median"),
            loan_rejected_pct=("loan_rejected", lambda s: 100 * s.mean()),
            disaster_5yr_pct=("disaster_5yr", lambda s: 100 * s.mean()),
            disaster_severe_pct=("disaster_severe_5yr", lambda s: 100 * s.mean()),
        )
        .round(2),
        "DEBUG",
    )


if __name__ == "__main__":
    main()
