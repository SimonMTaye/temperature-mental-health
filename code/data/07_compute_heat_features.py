"""Derive non-linear heat features from the existing daily temperature data.

Adds, per (kabupaten_code, date):
  hot28, hot30, hot32          1 if tmax_c crosses each threshold today
  hot28_7d, hot30_7d, hot32_7d count of crossings in past 7 days (incl today)
  hot28_30d, hot30_30d, hot32_30d                              past 30 days
  tmean_p90_30d                kab's local 90th-pct temperature over past 30d
  is_extreme                   1 if tmean_c > kab's all-time 95th-pct in our window
  cdd                          cooling-degree-days base 22°C   (max(t-22, 0))
  cdd_7d                       7-day rolling sum of CDD

Output appended to daily_temperature_kab.parquet (overwrites with extra columns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import OUT
from _schemas import DAILY_TEMPERATURE_HEAT_SCHEMA


def main() -> None:
    df = pd.read_parquet(OUT / "daily_temperature_kab.parquet")
    df = df.sort_values(["kabupaten_code", "date"]).reset_index(drop=True)
    g = df.groupby("kabupaten_code", group_keys=False)

    # daily threshold flags (use tmax for "hot day" definition — standard)
    for thr in [28, 30, 32]:
        df[f"hot{thr}"] = (df.tmax_c >= thr).astype(int)

    # rolling counts (closed on left so we don't peek at future)
    for thr in [28, 30, 32]:
        for win in [7, 30]:
            colname = f"hot{thr}_{win}d"
            df[colname] = (
                g[f"hot{thr}"]
                .rolling(window=win, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )

    # cooling degree days
    df["cdd"] = (df.tmean_c - 22).clip(lower=0)
    df["cdd_7d"] = (
        g["cdd"].rolling(window=7, min_periods=1).sum().reset_index(level=0, drop=True)
    )

    # local heat extremity (kab-level p90 of tmean across the window)
    p90 = g["tmean_c"].transform(lambda s: s.quantile(0.90))
    df["tmean_p90_30d"] = (
        g["tmean_c"].rolling(window=30, min_periods=15).quantile(0.90).reset_index(level=0, drop=True)
    )
    df["is_extreme"] = (df.tmean_c >= p90).astype(int)

    df = DAILY_TEMPERATURE_HEAT_SCHEMA.validate(df)
    df.to_parquet(OUT / "daily_temperature_kab.parquet", index=False)
    print(f"updated {OUT/'daily_temperature_kab.parquet'} with heat features")
    print(df[["hot28", "hot30", "hot32", "hot30_7d", "hot32_7d",
              "hot30_30d", "cdd_7d", "is_extreme"]].describe().round(2))


if __name__ == "__main__":
    main()
