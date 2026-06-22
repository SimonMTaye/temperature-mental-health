"""Clean community travel information from IFLS4 and IFLS5.

Output: data/generated/03_community_info.parquet
Row level: one community-wave record, keyed by community_id + wave.
"""

from pathlib import Path

import pandas as pd

from data._schemas import COMMUNITY_INFO_SCHEMA
from data._stata import read_stata_df
from data.config import GENERATED_DATA, RAW_IFLS_EXTRACTED
from library.log import log


PLACES = {
    "B": "market",
    "G": "district_capital",
    "H": "provincial_capital",
}
MEASURES = ["distance", "cost", "time", "method"]
OUTPUT_COLUMNS = [
    "community_id",
    "wave",
    *[
        f"community_travel_{place}_{measure}"
        for place in PLACES.values()
        for measure in MEASURES
    ],
]


def clean_wave(path: Path, community_column: str, wave: str) -> pd.DataFrame:
    df = read_stata_df(path, convert_categoricals=False)
    df = df[df["atype"].isin(PLACES)].copy()
    inside_village = df["a0"].eq(1)

    df["distance"] = pd.to_numeric(df["a1"], errors="coerce").mask(
        df["a1"].isin([998.98, 9999.98])
    )
    df["cost"] = pd.to_numeric(df["a4"], errors="coerce").mask(df["a4"].eq(999998))
    hours = pd.to_numeric(df["a3hr"], errors="coerce").mask(df["a3hr"].eq(98))
    minutes = pd.to_numeric(df["a3mnt"], errors="coerce").mask(df["a3mnt"].eq(98))
    df["time"] = hours * 60 + minutes
    df["method"] = pd.to_numeric(df["a2"], errors="coerce")

    df.loc[inside_village, ["distance", "cost", "time"]] = 0
    df.loc[inside_village, "method"] = pd.NA
    df["community_id"] = df[community_column].astype(str)
    df["wave"] = wave
    df["place"] = df["atype"].map(PLACES)
    return df[["community_id", "wave", "place", *MEASURES]]


def build_community_info() -> pd.DataFrame:
    waves = [
        clean_wave(
            RAW_IFLS_EXTRACTED / "IFLS4" / "cf07" / "bk1_a.dta",
            "commid07",
            "IFLS4",
        ),
        clean_wave(
            RAW_IFLS_EXTRACTED / "IFLS5" / "cf14" / "bk1_a1.dta",
            "commid14",
            "IFLS5",
        ),
    ]
    community_info = pd.concat(waves, ignore_index=True)
    community_info = community_info.pivot(
        index=["community_id", "wave"],
        columns="place",
        values=MEASURES,
    )
    community_info.columns = [
        f"community_travel_{place}_{measure}"
        for measure, place in community_info.columns
    ]
    community_info = community_info.reset_index()[OUTPUT_COLUMNS]
    assert len(community_info) == 624
    return COMMUNITY_INFO_SCHEMA.validate(community_info)


def main() -> None:
    community_info = build_community_info()
    output_path = GENERATED_DATA / "03_community_info.parquet"
    GENERATED_DATA.mkdir(parents=True, exist_ok=True)
    community_info.to_parquet(output_path, index=False)
    log(f"wrote {len(community_info):,} rows to {output_path}")


if __name__ == "__main__":
    main()
