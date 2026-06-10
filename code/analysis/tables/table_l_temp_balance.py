"""Table L: high-vs-low temperature balance on interview timing, demographics,
and the IFLS5 fuel-cut cash compensation.

Validity check for the heat design: identification comes from within
kecamatan-month-year weather wiggles, so the 7-day mean heat deviation
(``tmean_7d_dev``, the window the fuel-cut effect loads on) should be orthogonal
to who the respondents are and when they were interviewed. Interview timing and
demographics are balanced on the pooled IFLS4+IFLS5 sample the heat design runs
on; the fuel-cut variables -- including the BLT/BLSM cash transfer that
compensated the 2014 fuel-subsidy cut -- only exist in IFLS5 and are balanced
there. Each sample is split at its own median 7-day heat deviation. Per-row N
makes the pooled-vs-IFLS5 sample explicit. We report the high-minus-low
difference with kabupaten-clustered inference plus the normalized difference
(scale-free imbalance yardstick).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyfixest as pf

from _lettered_common import (
    CLUSTER,
    KECAMATAN_FE,
    fmt_count,
    load_analysis,
    require_columns,
    stars,
    write_outputs,
)

TABLE = "table_l_temp_balance"

# Identifying variation behind the fuel-cut result: the trailing 7-day mean
# temperature deviation from the local seasonal norm. Households are "high temp"
# if interviewed in a hotter-than-median window within their sample.
HEAT_SPLIT = "tmean_7d_dev"

# (panel title, [(row label, column)], sample) -- "pooled" uses both waves,
# "ifls5" restricts to the fuel-cut wave where the compensation variables exist.
PANELS = [
    (
        "A. Interview timing",
        [
            ("Interview month", "month"),
            ("Interview day-of-year", "iv_doy"),
            ("IFLS5 wave", "ifls5"),
        ],
        "pooled",
    ),
    (
        "B. Demographics",
        [
            ("Age", "age"),
            ("Female", "female"),
            ("Education (years)", "edu_yrs"),
            ("Married", "married"),
            ("Widowed", "widowed"),
            ("Household size", "hhsize"),
            ("Urban", "urban"),
            ("Log per-capita expenditure", "pce_log"),
            ("Urban vehicle household (baseline)", "urban_vehicle_hh_ifls4"),
        ],
        "pooled",
    ),
    (
        "C. Fuel-cut variables (IFLS5)",
        [
            ("Post-subsidy-cut interview", "post_subsidy"),
            ("Cash-transfer recipient", "cash_transfer_recipient"),
            ("BLT card holder", "blt_card"),
        ],
        "ifls5",
    ),
]


def prepare_sample(df: pd.DataFrame, sample: str) -> pd.DataFrame:
    """Restrict to the requested sample, add interview day-of-year, and split high/low temp."""
    require_columns(df, [HEAT_SPLIT, "interview_datetime", "wave"])
    out = df if sample == "pooled" else df[df["wave"] == "IFLS5"]
    out = out[out[HEAT_SPLIT].notna()].copy()
    out["iv_doy"] = pd.to_datetime(out["interview_datetime"]).dt.dayofyear
    threshold = out[HEAT_SPLIT].median()
    out["hot"] = (out[HEAT_SPLIT] > threshold).astype(int)
    return out


def balance_row(df: pd.DataFrame, variable: str) -> dict[str, float]:
    """Group means, sample size, and the high-minus-low difference with clustered inference."""
    frame = df.dropna(subset=[variable, CLUSTER]).copy()
    frame[variable] = pd.to_numeric(frame[variable], errors="coerce")
    frame = frame.dropna(subset=[variable])

    low = frame.loc[frame["hot"] == 0, variable]
    high = frame.loc[frame["hot"] == 1, variable]
    diff = float(high.mean() - low.mean())
    pooled_sd = np.sqrt((low.var() + high.var()) / 2)
    norm_diff = diff / pooled_sd if pooled_sd > 0 else np.nan

    model = pf.feols(f"{variable} ~ hot", data=frame, vcov={"CRV1": CLUSTER})
    p_value = float(model.pvalue().get("hot", np.nan))

    return {
        "low_mean": float(low.mean()),
        "high_mean": float(high.mean()),
        "diff": diff,
        "norm_diff": float(norm_diff),
        "p": p_value,
        "n_low": int(low.size),
        "n_high": int(high.size),
        "n": int(low.size + high.size),
    }


def main() -> None:
    base = load_analysis()
    all_vars = [variable for _, rows, _ in PANELS for _, variable in rows]
    # ``iv_doy`` is derived from interview_datetime inside prepare_sample.
    require_columns(base, [v for v in all_vars if v != "iv_doy"])
    samples = {name: prepare_sample(base, name) for name in ("pooled", "ifls5")}

    rows: list[dict[str, object]] = []
    body = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & Low temp & High temp & Diff & Norm. & \\",
        r"Variable & (mean) & (mean) & (High$-$Low) & diff & $N$ \\",
        r"\midrule",
    ]
    for panel, variables, sample in PANELS:
        data = samples[sample]
        body.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{panel}}}}} \\")
        for label, variable in variables:
            stats = balance_row(data, variable)
            rows.append({"panel": panel, "label": label, "var": variable, "sample": sample, **stats})
            body.append(
                rf"\quad {label} & {stats['low_mean']:.3f} & {stats['high_mean']:.3f} & "
                rf"${stats['diff']:+.3f}{stars(stats['p'])}$ & ${stats['norm_diff']:+.3f}$ & "
                rf"{fmt_count(stats['n'])} \\"
            )
        body.append(r"\addlinespace[3pt]")

    pooled = samples["pooled"]
    body.extend(
        [
            r"\midrule",
            rf"Kabupaten clusters & \multicolumn{{5}}{{c}}{{{pooled[CLUSTER].nunique():,}}} \\",
            rf"Kecamatan units & \multicolumn{{5}}{{c}}{{{pooled[KECAMATAN_FE].nunique():,}}} \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
