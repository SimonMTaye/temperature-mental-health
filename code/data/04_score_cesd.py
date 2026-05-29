"""Score the CES-D depression scale for IFLS4 and IFLS5.

Important note on IFLS4 item content: the IFLS4 hh07 codebook PDF claims the
kptype values A..J correspond to the older Radloff CES-D-20 short selection
(items including "appetite", "effort", "failure", "good as others", "talked less").
The actual MICRODATA at b3b_kp.dta, however, follows the Andresen 1994 CES-D-10
selection — verified empirically by item-level endorsement rates:

  Codebook claims:                   Endorsement rate    Plausible item:
    A = bothered (negative)              13% kp01=yes      ✓ bothered
    B = appetite (negative)              19%               ≈ trouble concentrating
    C = could not shake blues            14%               ≈ depressed
    D = good as others (positive)        35%               ≈ effort
    E = effort (negative)                89%               ✗ — too high for negative
                                                            ✓ hopeful (positive)
    F = hopeful (positive)               16%               ✗ — too low for positive
                                                            ✓ fearful (negative)
    G = failure (negative)               28%               ≈ restless sleep
    H = fearful (negative)               91%               ✗ — too high for negative
                                                            ✓ happy (positive)
    I = happy (positive)                 6%                ✗ — too low for positive
                                                            ✓ lonely (negative)
    J = talked less (negative)            8%               ≈ could not get going

The endorsement pattern unambiguously shows IFLS4 data follows the IFLS5 (Andresen)
mapping at letter positions. We therefore apply REVERSE_ITEMS = {E, H} to BOTH waves.
The §2.1 table in the long-form note (which followed the codebook PDF) was misleading
on item content — what's actually identical across the two waves is the 10-item set.

  Reverse-coded positive items in both waves: E = hopeful, H = happy

IFLS5 b3b_kp.dta — long format, kptype A..J × kp02 frequency 1..4.
IFLS4 b3b_kp.dta — same 10-item layout, with a SCREENER design:
    kp01 = "in past week did you feel [item]?" (1=yes, 3=no)
    kp02 = how often (only asked if kp01=1; otherwise NaN, treated as freq=0)

Outputs: data/generated/cesd_scores.parquet
  cols: pidlink, wave, n_items, cesd_raw (0-30 frequency score),
        cesd10_count (IFLS4 only — count of kp01=yes items),
        depressed (1 if cesd_raw>=10 — standard CES-D 10 cutoff)
"""
from __future__ import annotations

import pandas as pd

from config import OUT, RAW
from _schemas import CESD_SCORES_SCHEMA

REVERSE_ITEMS = {"E", "H"}  # hopeful, happy — applied to BOTH waves (see header docstring)


def score_ifls5() -> pd.DataFrame:
    df = pd.read_stata(RAW / "IFLS5/hh14/b3b_kp.dta", convert_categoricals=False)
    df = df[df.kp02.between(1, 4)]
    df = df.assign(score=df.kp02 - 1)
    df.loc[df.kptype.isin(REVERSE_ITEMS), "score"] = 3 - df.loc[df.kptype.isin(REVERSE_ITEMS), "score"]
    agg = df.groupby("pidlink").agg(n_items=("score", "size"), cesd_raw=("score", "sum")).reset_index()
    agg["wave"] = "IFLS5"
    agg["cesd10_count"] = float("nan")
    return agg


def score_ifls4() -> pd.DataFrame:
    df = pd.read_stata(RAW / "IFLS4/hh07/b3b_kp.dta", convert_categoricals=False)
    df = df[df.kp01.isin([1, 3])]  # 1=yes, 3=no
    df["yes"] = (df.kp01 == 1).astype(int)
    df.loc[df.kptype.isin(REVERSE_ITEMS), "yes"] = 1 - df.loc[df.kptype.isin(REVERSE_ITEMS), "yes"]

    df["freq"] = 0.0
    has_freq = df.kp01.eq(1) & df.kp02.between(1, 4)
    df.loc[has_freq, "freq"] = df.loc[has_freq, "kp02"] - 1
    df.loc[df.kp01.eq(1) & df.kp02.isna(), "freq"] = 1.5
    df.loc[df.kptype.isin(REVERSE_ITEMS), "freq"] = 3 - df.loc[df.kptype.isin(REVERSE_ITEMS), "freq"]

    agg = df.groupby("pidlink").agg(
        n_items=("yes", "size"),
        cesd10_count=("yes", "sum"),
        cesd_raw=("freq", "sum"),
    ).reset_index()
    agg["wave"] = "IFLS4"
    return agg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [score_ifls4(), score_ifls5()]
    out = pd.concat(parts, ignore_index=True)
    out["depressed"] = (out.cesd_raw >= 10).astype(int)
    out = out[["pidlink", "n_items", "cesd_raw", "wave", "cesd10_count", "depressed"]]
    out = CESD_SCORES_SCHEMA.validate(out)
    out.to_parquet(OUT / "cesd_scores.parquet", index=False)
    print(f"wrote {len(out):,} rows to {OUT/'cesd_scores.parquet'}")
    print(out.groupby("wave").agg(
        n=("pidlink", "size"),
        n_items_med=("n_items", "median"),
        cesd_raw_mean=("cesd_raw", "mean"),
        cesd_raw_p50=("cesd_raw", "median"),
        depressed_pct=("depressed", lambda x: 100*x.mean()),
    ).round(2))


if __name__ == "__main__":
    main()
