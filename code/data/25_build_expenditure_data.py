"""Build household expenditure and transport-share measures.

Output: data/generated/25_expenditure_data.parquet
Row level: one record per (pidlink, wave), using the individual panel skeleton
from 01_individuals.parquet.
"""

import numpy as np
import pandas as pd

from data._schemas import EXPENDITURE_DATA_SCHEMA  # noqa: E402
from data._sentinels import clean_money  # noqa: E402
from data._stata import read_stata_df  # noqa: E402
from data.config import GENERATED_DATA, IFLS4_FOLDER, IFLS5_FOLDER  # noqa: E402
from library.log import log  # noqa: E402


SPENDING_QUANTILE_COLUMNS = [
    "fuel_total",
    "transport_total",
    "fuel_transport_total",
    "fuel_share",
    "transport_share",
    "fuel_transport_share",
]

IFLS_FOLDERS = {
    "IFLS4": IFLS4_FOLDER,
    "IFLS5": IFLS5_FOLDER,
}

HHID_COLUMNS = {
    "IFLS4": "hhid07",
    "IFLS5": "hhid14",
}

MONTHS_PER_YEAR = 12
WEEKS_PER_MONTH = 4.33
COOKING_OIL_FOOD_TYPE = "Y"


def _grouped_money_sum(
    df: pd.DataFrame,
    *,
    hhid_column: str,
    value_column: str,
    output_column: str,
) -> pd.DataFrame:
    out = df[[hhid_column, value_column]].copy()
    out[value_column] = clean_money(out[value_column])
    return (
        out.groupby(hhid_column)[value_column]
        .sum(min_count=1)
        .rename(output_column)
        .reset_index()
    )


def _sum_clean_money_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    cleaned = pd.DataFrame({column: clean_money(df[column]) for column in columns})
    return cleaned.sum(axis=1, min_count=1)


def food_expenditure(
    ks1: pd.DataFrame,
    hhid_column: str,
) -> pd.DataFrame:
    """Build local monthly food-expenditure components from KS1."""
    food = ks1[[hhid_column, "ks1type", "ks02"]].copy()
    food["ks02"] = clean_money(food["ks02"])
    food["expenditure_food_items_mo"] = food["ks02"].where(
        food["ks1type"] != COOKING_OIL_FOOD_TYPE,
        0,
    )
    food["expenditure_food_cooking_oil_mo"] = food["ks02"].where(
        food["ks1type"] == COOKING_OIL_FOOD_TYPE,
        0,
    )

    out = (
        food.groupby(hhid_column)[
            ["expenditure_food_items_mo", "expenditure_food_cooking_oil_mo"]
        ]
        .sum(min_count=1)
        .mul(WEEKS_PER_MONTH)
        .reset_index()
        .assign(
            expenditure_food_total_mo=lambda df: df[
                ["expenditure_food_items_mo", "expenditure_food_cooking_oil_mo"]
            ].sum(axis=1, min_count=1)
        )
    )
    return out.rename(columns={hhid_column: "hhid"})


def non_food_expenditure(
    ks0: pd.DataFrame,
    ks2: pd.DataFrame,
    ks3: pd.DataFrame,
    hhid_column: str,
) -> pd.DataFrame:
    """Build local monthly non-food expenditure components from KS modules."""
    ks2_total = _grouped_money_sum(
        ks2,
        hhid_column=hhid_column,
        value_column="ks06",
        output_column="expenditure_nonfood_ks2_mo",
    )

    ks2_component = ks2[[hhid_column, "ks2type", "ks06"]].copy()
    ks2_component["ks06"] = clean_money(ks2_component["ks06"])
    ks2_component = (
        ks2_component[ks2_component["ks2type"].isin(["A3", "E"])]
        .pivot_table(
            index=hhid_column,
            columns="ks2type",
            values="ks06",
            aggfunc=lambda values: values.sum(min_count=1),
        )
        .rename(
            columns={
                "A3": "expenditure_nonfood_fuel_mo",
                "E": "expenditure_nonfood_transport_mo",
            }
        )
        .reset_index()
    )
    CHILDREN_EDUCATION_EXPENDITURE_COLUMNS = [
        "ks10aa",
        "ks10ab",
        "ks11aa",
        "ks11ab",
        "ks12aa",
        "ks12ab",
    ]
    ks0_education = ks0[[hhid_column, *CHILDREN_EDUCATION_EXPENDITURE_COLUMNS]].copy()
    ks0_education["expenditure_nonfood_children_education_mo"] = (
        _sum_clean_money_columns(ks0_education, CHILDREN_EDUCATION_EXPENDITURE_COLUMNS)
        / MONTHS_PER_YEAR
    )
    ks0_education = (
        ks0_education.groupby(hhid_column)["expenditure_nonfood_children_education_mo"]
        .sum(min_count=1)
        .reset_index()
    )

    ks0_food_gift = _grouped_money_sum(
        ks0,
        hhid_column=hhid_column,
        value_column="ks04b",
        output_column="expenditure_nonfood_food_gift_mo",
    )
    # Convert to monthly from weekly
    ks0_food_gift["expenditure_nonfood_food_gift_mo"] *= WEEKS_PER_MONTH

    ks3_total = _grouped_money_sum(
        ks3,
        hhid_column=hhid_column,
        value_column="ks08",
        output_column="expenditure_nonfood_ks3_mo",
    )
    ks3_total["expenditure_nonfood_ks3_mo"] /= MONTHS_PER_YEAR

    out = (
        ks2_total.merge(ks2_component, on=hhid_column, how="left", validate="1:1")
        .merge(ks0_education, on=hhid_column, how="left", validate="1:1")
        .merge(ks0_food_gift, on=hhid_column, how="left", validate="1:1")
        .merge(ks3_total, on=hhid_column, how="left", validate="1:1")
        .assign(
            expenditure_nonfood_kerosene_mo=np.nan,
            expenditure_nonfood_total_mo=lambda df: df[
                [
                    "expenditure_nonfood_ks2_mo",
                    "expenditure_nonfood_children_education_mo",
                    "expenditure_nonfood_food_gift_mo",
                    "expenditure_nonfood_ks3_mo",
                ]
            ].sum(axis=1, min_count=1),
        )
    )
    return out.rename(columns={hhid_column: "hhid"})


def _main_crop(wave: str) -> pd.DataFrame:
    """Read the household's most valuable crop or livestock from IFLS b2_ut1."""
    folder = IFLS_FOLDERS[wave]
    hhid_column = HHID_COLUMNS[wave]
    ut1 = read_stata_df(folder / "b2_ut1.dta", convert_categoricals=False)
    out = ut1[[hhid_column, "ut07a"]].rename(
        columns={hhid_column: "hhid", "ut07a": "main_crop"}
    )
    out["main_crop"] = pd.to_numeric(out["main_crop"], errors="coerce")
    out["wave"] = wave
    return out


def _add_spending_quantiles(out: pd.DataFrame) -> pd.DataFrame:
    for column in SPENDING_QUANTILE_COLUMNS:
        out[f"{column}_quartile"] = out.groupby("wave")[column].transform(
            lambda series: pd.qcut(series.rank(method="first"), 4, labels=False) + 1,
        )
        out[f"{column}_quintile"] = out.groupby("wave")[column].transform(
            lambda series: pd.qcut(series.rank(method="first"), 5, labels=False) + 1,
        )
    out["transport_share_q5"] = out["transport_share_quintile"]
    out["high_transport_share"] = out.transport_share_quintile == 5
    return out


def expenditure_data() -> pd.DataFrame:
    """Build expenditure-related measures from IFLS modules."""
    frame = pd.DataFrame()
    for wave in ["IFLS4", "IFLS5"]:
        p = IFLS_FOLDERS[wave] / "b1_ks0.dta"
        if not p.exists():
            continue
        ks0 = read_stata_df(p, convert_categoricals=False)
        hhid_col = HHID_COLUMNS[wave]
        ks1 = read_stata_df(
            IFLS_FOLDERS[wave] / "b1_ks1.dta", convert_categoricals=False
        )
        ks2 = read_stata_df(
            IFLS_FOLDERS[wave] / "b1_ks2.dta", convert_categoricals=False
        )
        ks3 = read_stata_df(
            IFLS_FOLDERS[wave] / "b1_ks3.dta", convert_categoricals=False
        )
        food = food_expenditure(ks1, hhid_column=hhid_col)
        non_food = non_food_expenditure(ks0, ks2, ks3, hhid_column=hhid_col)
        expenditure = food.merge(non_food, on=["hhid"], how="outer", validate="1:1")
        expenditure["expenditure_total_mo"] = expenditure[
            ["expenditure_food_total_mo", "expenditure_nonfood_total_mo"]
        ].sum(axis=1, min_count=1)
        expenditure["wave"] = wave
        frame = pd.concat([frame, expenditure], ignore_index=True)
    return frame


def compute_transport_share(expenditure: pd.DataFrame) -> pd.DataFrame:
    expenditure["transport_total"] = expenditure[
        ["expenditure_nonfood_transport_mo"]
    ].sum(axis=1, min_count=1)

    expenditure["fuel_total"] = expenditure[["expenditure_nonfood_fuel_mo"]].sum(
        axis=1, min_count=1
    )
    expenditure["fuel_transport_total"] = expenditure[
        ["transport_total", "fuel_total"]
    ].sum(axis=1, min_count=1)
    expenditure["transport_spending_mo"] = expenditure.transport_total
    expenditure["total_mo"] = expenditure[
        ["expenditure_nonfood_total_mo", "expenditure_food_total_mo"]
    ].sum(axis=1, min_count=1)

    expenditure["transport_share"] = (
        expenditure.transport_total / expenditure.total_mo.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    expenditure["fuel_share"] = (
        expenditure.fuel_total / expenditure.total_mo.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    expenditure["fuel_transport_share"] = (
        expenditure.fuel_transport_total / expenditure.total_mo.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    # Compute z-scores of the shares
    for share_column in ["transport_share", "fuel_share", "fuel_transport_share"]:
        expenditure[f"{share_column}_z"] = (
            expenditure[share_column] - expenditure[share_column].mean()
        ) / expenditure[share_column].std()
    return expenditure


def main() -> pd.DataFrame:
    """Build and write the 25-prefixed expenditure sidecar."""
    expenditure = expenditure_data().pipe(compute_transport_share)

    main_crop = pd.concat(
        [_main_crop("IFLS4"), _main_crop("IFLS5")],
        ignore_index=True,
    )
    log(
        f"expenditure rows: {len(expenditure):,}; "
        f"median transport share={expenditure.transport_share.median():.3f}"
    )

    out_final = (
        pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")
        .loc[:, ["pidlink", "wave", "hhid", "province_code"]]
        .drop_duplicates(["pidlink", "wave"])
        .merge(
            expenditure.drop_duplicates(["hhid", "wave"]),
            on=["hhid", "wave"],
            how="left",
            validate="m:1",
        )
        .merge(
            main_crop.drop_duplicates(["hhid", "wave"]),
            on=["hhid", "wave"],
            how="left",
            validate="m:1",
        )
        .pipe(_add_spending_quantiles)
    )
    out_final = out_final[list(EXPENDITURE_DATA_SCHEMA.columns)]
    out_final = EXPENDITURE_DATA_SCHEMA.validate(out_final)

    output_path = GENERATED_DATA / "25_expenditure_data.parquet"
    out_final.to_parquet(output_path, index=False)

    log(f"wrote {len(out_final):,} rows to {output_path}")
    log("commodity/transport prevalence by wave:", "DEBUG")
    log(
        out_final.groupby("wave")
        .agg(
            n=("pidlink", "size"),
            high_transport_pct=("high_transport_share", lambda s: 100 * s.mean()),
            transport_share_med=("transport_share", "median"),
        )
        .round(3),
        "DEBUG",
    )
    return out_final


if __name__ == "__main__":
    main()
