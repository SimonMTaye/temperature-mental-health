"""Build IFLS5 sleep-duration outcome from time-use wake and sleep times.

Output: data/generated/28_sleep_duration.parquet
Row level: one IFLS5 person-wave record, keyed by pidlink + wave.
"""

import numpy as np
import pandas as pd

from _schemas import SLEEP_DURATION_SCHEMA
from _stata import read_stata_df
from config import GENERATED_DATA, IFLS5_FOLDER
from log import log


def build_sleep_duration() -> pd.DataFrame:
    df = read_stata_df(IFLS5_FOLDER / "b3a_pna1.dta", convert_categoricals=False)
    start_n = len(df)
    required = ["pidlink", "pna04hr", "pna04mnt", "pna05hr", "pna5mnt"]
    missing_time = df[required].isna().any(axis=1)
    log(f"sleep duration: dropping {missing_time.sum():,} rows with missing time fields")
    df = df[~missing_time].copy()

    valid_wake = df.pna04hr.between(0, 23) & df.pna04mnt.between(0, 59)
    log(f"sleep duration: dropping {(~valid_wake).sum():,} rows with invalid wake times")
    df = df[valid_wake].copy()

    valid_sleep = df.pna05hr.between(0, 23) & df.pna5mnt.between(0, 59)
    log(f"sleep duration: dropping {(~valid_sleep).sum():,} rows with invalid sleep times")
    df = df[valid_sleep].copy()

    wake_h = df.pna04hr + df.pna04mnt / 60.0
    sleep_h = df.pna05hr + df.pna5mnt / 60.0
    df["sleep_dur_h"] = np.where(
        sleep_h >= 18,
        (24.0 - sleep_h) + wake_h,
        wake_h - sleep_h,
    )
    valid_duration = df.sleep_dur_h.between(0.5, 16.0)
    log(
        f"sleep duration: dropping {(~valid_duration).sum():,} rows with duration outside 0.5-16h"
    )
    df = df[valid_duration].copy()
    out = df[["pidlink", "sleep_dur_h"]].drop_duplicates("pidlink").copy()
    out["wave"] = "IFLS5"
    out = out[["pidlink", "wave", "sleep_dur_h"]]
    log(f"sleep duration: kept {len(out):,} people from {start_n:,} raw rows")
    return SLEEP_DURATION_SCHEMA.validate(out)


def main() -> None:
    out = build_sleep_duration()
    output_path = GENERATED_DATA / "28_sleep_duration.parquet"
    out.to_parquet(output_path, index=False)
    log(f"wrote {len(out):,} rows to {output_path}")


if __name__ == "__main__":
    main()
