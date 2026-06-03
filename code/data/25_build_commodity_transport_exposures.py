"""Build commodity-region farmer and household transport-share exposures.

Output: data/generated/25_commodity_transport_exposures.parquet
Row level: one record per (pidlink, wave), using the individual panel skeleton
from 01_individuals.parquet.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _schemas import COMMODITY_TRANSPORT_EXPOSURES_SCHEMA  # noqa: E402
from _sentinels import clean_money  # noqa: E402
from _stata import read_stata_df  # noqa: E402
from config import GENERATED_DATA, IFLS4_FOLDER, IFLS5_FOLDER  # noqa: E402
from log import log  # noqa: E402


PALM_PROVS = {
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    21,
    61,
    62,
    63,
    64,
}

# Top rubber provinces: North Sumatra, Riau, Jambi, South Sumatra, West Kalimantan.
RUBBER_PROVS = {12, 14, 15, 16, 61}

# Top coffee provinces: Aceh, North Sumatra, South Sumatra, Lampung, Bali, Sulawesi.
COFFEE_PROVS = {11, 12, 16, 18, 51, 73}

OUTPUT_COLUMNS = [
    "pidlink",
    "wave",
    "agricultural",
    "palm_region",
    "palm_farmer_individual",
    "palm_farmer_hh",
    "rubber_region",
    "rubber_farmer_individual",
    "coffee_region",
    "coffee_farmer_individual",
    "transport_spending_mo",
    "total_mo",
    "transport_share",
    "transport_share_q5",
    "high_transport_share",
]

IFLS_FOLDERS = {
    "IFLS4": IFLS4_FOLDER,
    "IFLS5": IFLS5_FOLDER,
}

HHID_COLUMNS = {
    "IFLS4": "hhid07",
    "IFLS5": "hhid14",
}


def add_worker_sector_dummy(wave: str) -> pd.DataFrame:
    tk = read_stata_df(
        IFLS_FOLDERS[wave] / "b3a_tk2.dta",
        convert_categoricals=False,
    )
    assert "pidlink" in tk.columns, f"Expected pidlink column in {wave} tk dataset"
    assert "tk19ab" in tk.columns, f"Expected pidlink column in {wave} tk dataset"
    sector = pd.to_numeric(tk.tk19ab.astype(str).str.strip())
    out = tk[["pidlink"]].copy()
    out["agricultural"] = sector == 1
    out["mining"] = sector == 2
    out["wave"] = wave
    return out


def _monthly_food_spending(ks0: pd.DataFrame, *, hhid_col_name: str) -> pd.DataFrame:
    """Convert weekly food spending to monthly household spending."""
    if "ks02a" in ks0.columns:
        ks0 = ks0.copy()
        # We use 4.33 to convert weekly to monthly spending
        ks0["food_mo"] = clean_money(ks0.ks02a) * 4.33
    else:
        ks0 = pd.DataFrame({hhid_col_name: ks0[hhid_col_name], "food_mo": np.nan})
    return ks0[[hhid_col_name, "food_mo"]]


def _transport_share_from_frames(
    ks2: pd.DataFrame,
    ks3: pd.DataFrame,
    ks0: pd.DataFrame,
    *,
    hhid_col_name: str,
    wave: str,
) -> pd.DataFrame:
    """Compute transportation share of monthly household spending."""
    ks2 = ks2.copy()
    ks3 = ks3.copy()
    ks2["ks06"] = clean_money(ks2.ks06)
    ks3["ks08"] = clean_money(ks3.ks08)
    transport = ks2[ks2.ks2type == "E"][[hhid_col_name, "ks06"]].rename(
        columns={"ks06": "transport_spending_mo"}
    )

    total_ks2 = (
        ks2.groupby(hhid_col_name)["ks06"]
        .sum(min_count=1)
        .rename("total_ks2_mo")
        .reset_index()
    )
    total_ks3 = (
        ks3.groupby(hhid_col_name)["ks08"]
        .sum(min_count=1)
        .rename("total_ks3_mo")
        .reset_index()
    )
    food = _monthly_food_spending(ks0, hhid_col_name=hhid_col_name)

    out = transport.drop_duplicates(hhid_col_name).merge(
        total_ks2, on=hhid_col_name, validate="1:1"
    )
    out = out.merge(total_ks3, on=hhid_col_name, validate="1:1").merge(
        food,
        on=hhid_col_name,
        how="left",
        validate="1:1",
    )
    if wave == "IFLS5":
        for col in ["transport_spending_mo", "total_ks2_mo", "total_ks3_mo", "food_mo"]:
            out[col] = out[col].fillna(0)
    out["total_mo"] = out.total_ks2_mo + out.total_ks3_mo + out.food_mo
    out["transport_share"] = out.transport_spending_mo / out.total_mo.replace(0, np.nan)
    out = out[(out.transport_share >= 0) & (out.transport_share <= 1)]
    out = out.rename(columns={hhid_col_name: "hhid"})[
        ["hhid", "transport_spending_mo", "total_mo", "transport_share"]
    ]
    out["wave"] = wave
    return out


def _transport_share(wave: str) -> pd.DataFrame:
    folder = IFLS_FOLDERS[wave]
    return _transport_share_from_frames(
        read_stata_df(folder / "b1_ks2.dta", convert_categoricals=False),
        read_stata_df(folder / "b1_ks3.dta", convert_categoricals=False),
        read_stata_df(folder / "b1_ks0.dta", convert_categoricals=False),
        hhid_col_name=HHID_COLUMNS[wave],
        wave=wave,
    )


def _add_region_worker_flags(out: pd.DataFrame) -> pd.DataFrame:
    is_agricultural = out.agricultural.fillna(0) == 1
    out["palm_region"] = out.province_code.isin(PALM_PROVS)
    out["palm_farmer_individual"] = is_agricultural & out.palm_region
    out["palm_farmer_hh"] = out.groupby(["hhid", "wave"])[
        "palm_farmer_individual"
    ].transform("max")
    out["rubber_region"] = out.province_code.isin(RUBBER_PROVS)
    out["rubber_farmer_individual"] = is_agricultural & out.rubber_region
    out["coffee_region"] = out.province_code.isin(COFFEE_PROVS)
    out["coffee_farmer_individual"] = is_agricultural & out.coffee_region
    return out


def _add_transport_quintile(out: pd.DataFrame) -> pd.DataFrame:
    out["transport_share_q5"] = out.groupby("wave")["transport_share"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1
    )
    out["high_transport_share"] = out.transport_share_q5 == 5
    return out


def _finalize_output(out: pd.DataFrame) -> pd.DataFrame:
    out_final = out[OUTPUT_COLUMNS].copy()
    for col in [
        "palm_region",
        "palm_farmer_individual",
        "palm_farmer_hh",
        "rubber_region",
        "rubber_farmer_individual",
        "coffee_region",
        "coffee_farmer_individual",
        "high_transport_share",
    ]:
        out_final[col] = out_final[col].fillna(0).astype(int)
    if "agricultural" in out_final.columns:
        out_final["agricultural"] = out_final.agricultural.fillna(0).astype(int)
    return COMMODITY_TRANSPORT_EXPOSURES_SCHEMA.validate(out_final)


def build_commodity_transport_exposures() -> pd.DataFrame:
    """Build and write the 25-prefixed commodity/transport sidecar."""
    agricultural = pd.concat(
        [add_worker_sector_dummy("IFLS4"), add_worker_sector_dummy("IFLS5")],
        ignore_index=True,
    )
    transport = pd.concat(
        [_transport_share("IFLS4"), _transport_share("IFLS5")],
        ignore_index=True,
    )
    log(
        f"sector rows: {len(agricultural):,}; transport rows: {len(transport):,}; "
        f"median transport share={transport.transport_share.median():.3f}"
    )

    individuals = pd.read_parquet(GENERATED_DATA / "01_individuals.parquet")
    base = individuals[["pidlink", "wave", "hhid"]].drop_duplicates(["pidlink", "wave"])
    out = base.merge(
        individuals[["pidlink", "wave", "province_code"]],
        on=["pidlink", "wave"],
        how="left",
        validate="1:1",
    )
    out = out.merge(agricultural, on=["pidlink", "wave"], how="left", validate="1:1")
    out = out.merge(
        transport.drop_duplicates(subset=["hhid", "wave"]),
        on=["hhid", "wave"],
        how="left",
        validate="m:1",
    )
    out = _add_region_worker_flags(out)
    ifls4_worker_dummies = out[out.wave == "IFLS4"][
        [
            "pidlink",
            "agricultural",
            "palm_farmer_individual",
            "palm_farmer_hh",
        ]
    ].copy()
    # add _ifsl4 suffix to columns
    ifls4_worker_dummies = ifls4_worker_dummies.rename(
        columns={
            "agricultural": "agricultural_ifls4",
            "palm_farmer_individual": "palm_farmer_individual_ifls4",
            "palm_farmer_hh": "palm_farmer_hh_ifls4",
        }
    )
    out = out.merge(ifls4_worker_dummies, on="pidlink", how="left", validate="m:1")
    out = _add_transport_quintile(out)
    out_final = _finalize_output(out)
    output_path = GENERATED_DATA / "25_commodity_transport_exposures.parquet"
    out_final.to_parquet(output_path, index=False)

    log(f"wrote {len(out_final):,} rows to {output_path}")
    log("commodity/transport prevalence by wave:", "DEBUG")
    log(
        out_final.groupby("wave")
        .agg(
            n=("pidlink", "size"),
            agricultural_pct=("agricultural", lambda s: 100 * s.fillna(0).mean()),
            palm_farmer_pct=("palm_farmer_individual", lambda s: 100 * s.mean()),
            rubber_farmer_pct=("rubber_farmer_individual", lambda s: 100 * s.mean()),
            coffee_farmer_pct=("coffee_farmer_individual", lambda s: 100 * s.mean()),
            high_transport_pct=("high_transport_share", lambda s: 100 * s.mean()),
            transport_share_med=("transport_share", "median"),
        )
        .round(3),
        "DEBUG",
    )
    return out_final


def main() -> None:
    build_commodity_transport_exposures()


if __name__ == "__main__":
    main()
