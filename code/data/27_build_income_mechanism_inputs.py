"""Build household income mechanism outcomes for analysis tables.

Output: data/generated/27_income_mechanism_inputs.parquet
Row level: one person-wave record, keyed by pidlink + wave.
"""

import numpy as np
import pandas as pd

from _schemas import INCOME_MECHANISM_INPUTS_SCHEMA
from _sentinels import clean_money
from _stata import read_stata_df
from config import GENERATED_DATA, IDR_2007_TO_2014_DEFLATOR, IFLS4_FOLDER, IFLS5_FOLDER
from log import log


IFLS_FOLDERS = {
    "IFLS4": IFLS4_FOLDER,
    "IFLS5": IFLS5_FOLDER,
}

HHID_COLUMNS = {
    "IFLS4": "hhid07",
    "IFLS5": "hhid14",
}


def _sum_existing(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return pd.Series(0.0, index=df.index)
    return (
        pd.concat([clean_money(df[col]) for col in existing], axis=1)
        .sum(axis=1, min_count=1)
        .fillna(0)
    )


def _labor_income(wave: str) -> pd.DataFrame:
    tk = read_stata_df(IFLS_FOLDERS[wave] / "b3a_tk2.dta", convert_categoricals=False)
    income_cols = ["tk25a1", "tk25b1", "tk26a1", "tk26b1"]
    out = tk[["pidlink"]].copy()
    out["person_labor_income_mo"] = _sum_existing(tk, income_cols)
    out["wave"] = wave
    out = out.groupby(["pidlink", "wave"], as_index=False)["person_labor_income_mo"].sum()
    skeleton = pd.read_parquet(GENERATED_DATA / "01_individuals.parquet").loc[
        lambda df: df.wave == wave, ["pidlink", "hhid", "wave"]
    ]
    out = skeleton.merge(out, on=["pidlink", "wave"], how="left", validate="1:1")
    out["person_labor_income_mo"] = out.person_labor_income_mo.fillna(0)
    return (
        out.groupby(["hhid", "wave"], as_index=False)["person_labor_income_mo"]
        .sum()
        .rename(columns={"person_labor_income_mo": "hh_labor_income_mo"})
    )


def _annual_household_profit(
    wave: str, filename: str, value_columns: list[str]
) -> pd.DataFrame:
    path = IFLS_FOLDERS[wave] / filename
    if not path.exists():
        return pd.DataFrame(columns=["hhid", "wave", "annual_profit"])
    df = read_stata_df(path, convert_categoricals=False)
    hhid_col = HHID_COLUMNS[wave]
    out = pd.DataFrame({"hhid": df[hhid_col]})
    out["annual_profit"] = _sum_existing(df, value_columns)
    out["wave"] = wave
    return out.groupby(["hhid", "wave"], as_index=False)["annual_profit"].sum()


def _nonlabor_income(wave: str) -> pd.DataFrame:
    frames = [
        _annual_household_profit(wave, "b2_ut1.dta", ["ut09"]),
        _annual_household_profit(wave, "b2_nt2.dta", ["nt09", "nt26"]),
    ]
    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        return pd.DataFrame(columns=["hhid", "wave", "hh_nonlabor_income_mo"])
    out = out.groupby(["hhid", "wave"], as_index=False)["annual_profit"].sum()
    out["hh_nonlabor_income_mo"] = out.annual_profit / 12.0
    return out[["hhid", "wave", "hh_nonlabor_income_mo"]]


def _winsorize_within_wave(s: pd.Series, wave: pd.Series) -> pd.Series:
    def trim(x: pd.Series) -> pd.Series:
        if x.dropna().empty:
            return x
        return x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))

    return s.groupby(wave).transform(trim)


def build_income_mechanism_inputs() -> pd.DataFrame:
    skeleton = pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")[
        ["pidlink", "hhid", "wave"]
    ].drop_duplicates(["pidlink", "wave"])
    labor = pd.concat([_labor_income("IFLS4"), _labor_income("IFLS5")], ignore_index=True)
    nonlabor = pd.concat([_nonlabor_income("IFLS4"), _nonlabor_income("IFLS5")], ignore_index=True)
    transport = pd.read_parquet(GENERATED_DATA / "25_commodity_transport_exposures.parquet")[
        ["pidlink", "wave", "transport_spending_mo", "transport_share"]
    ]

    out = (
        skeleton.merge(labor, on=["hhid", "wave"], how="left", validate="m:1")
        .merge(nonlabor, on=["hhid", "wave"], how="left", validate="m:1")
        .merge(transport, on=["pidlink", "wave"], how="left", validate="1:1")
    )
    out["hh_labor_income_mo"] = out.hh_labor_income_mo.fillna(0)
    out["hh_nonlabor_income_mo"] = out.hh_nonlabor_income_mo.fillna(0)
    out["income_deflator"] = np.where(out.wave == "IFLS4", IDR_2007_TO_2014_DEFLATOR, 1.0)
    out["labor_real"] = out.hh_labor_income_mo * out.income_deflator
    out["nonlabor_real"] = out.hh_nonlabor_income_mo * out.income_deflator
    out["transport_real"] = out.transport_spending_mo * out.income_deflator
    out["labor_real_w"] = _winsorize_within_wave(out.labor_real, out.wave)
    out["nonlabor_real_w"] = _winsorize_within_wave(out.nonlabor_real, out.wave)
    out["transport_real_w"] = _winsorize_within_wave(out.transport_real, out.wave)
    out["transport_share_ihs"] = np.arcsinh(out.transport_share)

    out = out[list(INCOME_MECHANISM_INPUTS_SCHEMA.columns)]
    return INCOME_MECHANISM_INPUTS_SCHEMA.validate(out)


def main() -> None:
    out = build_income_mechanism_inputs()
    output_path = GENERATED_DATA / "27_income_mechanism_inputs.parquet"
    out.to_parquet(output_path, index=False)
    log(f"wrote {len(out):,} rows to {output_path}")


if __name__ == "__main__":
    main()
