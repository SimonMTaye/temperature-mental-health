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

Outputs:
  data/generated/24_cesd_scores_loose.parquet:
        all scoreable person-wave rows, including incomplete CES-D rows.
  data/generated/24_cesd_scores.parquet:
        complete 10-item person-wave rows validated against the CES-D schema.

  cols: pidlink, wave, cesd_raw (0-30 frequency score),
        cesd10_count (IFLS4 only — count of kp01=yes items),
        depressed (1 if cesd_raw>=10 — standard CES-D 10 cutoff),
        n_items, CES-D factor scores, and within-wave factor z-scores
"""

import pandas as pd

from config import GENERATED_DATA, RAW_IFLS_EXTRACTED
from _schemas import CESD_SCORES_SCHEMA
from _stata import read_stata_df
from log import log

REVERSE_ITEMS = {"E", "H"}
CESD_ITEMS = set("ABCDEFGHIJ")
FACTOR_MAP = {
    "somatic": {"A", "B", "D", "G", "J"},
    "depraffect": {"C", "F", "I"},
    "posaffect": {"E", "H"},
}
OUTPUT_COLUMNS = [
    "pidlink",
    "wave",
    "cesd_raw",
    "cesd10_count",
    "depressed",
    "n_items",
    "somatic",
    "depraffect",
    "posaffect",
    "somatic_z",
    "depraffect_z",
    "posaffect_z",
]
COMPLETE_SCORE_COLUMNS = [
    "cesd_raw",
    "somatic",
    "depraffect",
    "posaffect",
    "somatic_z",
    "depraffect_z",
    "posaffect_z",
]


def score_item_frame(df: pd.DataFrame, wave: str) -> pd.DataFrame:
    """Return long item scores with the IFLS4 screener and IFLS5 frequency rules."""
    if wave == "IFLS5":
        df = df[df.kp02.between(1, 4)].copy()
        df["score"] = df.kp02 - 1
    else:
        df = df[df.kp01.isin([1, 3])].copy()
        df["score"] = 0.0
        has_freq = df.kp01.eq(1) & df.kp02.between(1, 4)
        df.loc[has_freq, "score"] = df.loc[has_freq, "kp02"] - 1
        df.loc[df.kp01.eq(1) & df.kp02.isna(), "score"] = 1.5

    rev = df.kptype.isin(REVERSE_ITEMS)
    df.loc[rev, "score"] = 3 - df.loc[rev, "score"]
    guard_scored_items(df)
    return df[["pidlink", "kptype", "score"]]


def guard_scored_items(df: pd.DataFrame) -> None:
    """Fail before aggregation if item codes or scored values leave CES-D bounds."""
    bad_items = sorted(set(df.kptype.dropna()) - CESD_ITEMS)
    if bad_items:
        raise ValueError(f"unexpected CES-D item codes: {bad_items}")
    if not df.score.between(0, 3).all():
        bad = df.loc[~df.score.between(0, 3), ["pidlink", "kptype", "score"]].head()
        raise ValueError(f"CES-D item scores outside 0..3:\n{bad}")


def build_factor_scores(items: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scored CES-D items to factor scores."""
    factors = []
    for factor_name, item_set in FACTOR_MAP.items():
        scores = (
            items[items.kptype.isin(item_set)]
            .groupby(["pidlink", "wave"])["score"]
            .sum()
            .rename(factor_name)
            .reset_index()
        )
        factors.append(scores)

    out = factors[0]
    for factor in factors[1:]:
        out = out.merge(factor, on=["pidlink", "wave"], how="outer", validate="1:1")
    return out


def add_factor_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add within-wave standardized factor scores."""
    out = df.copy()
    for factor_name in FACTOR_MAP:
        out[f"{factor_name}_z"] = out.groupby("wave")[factor_name].transform(
            lambda s: (s - s.mean()) / s.std()
        )
    return out


def build_output(
    score_parts: list[pd.DataFrame], item_parts: list[pd.DataFrame]
) -> pd.DataFrame:
    """Build person-wave CES-D scores from wave-level score and item frames."""
    out = pd.concat(score_parts, ignore_index=True)
    factors = build_factor_scores(pd.concat(item_parts, ignore_index=True))
    out = out.merge(factors, on=["pidlink", "wave"], how="left", validate="1:1")
    out["depressed"] = (out.cesd_raw >= 10).astype(int)
    out = add_factor_z_scores(out)
    return out[OUTPUT_COLUMNS]


def complete_only(loose: pd.DataFrame) -> pd.DataFrame:
    """Return canonical CES-D rows with all 10 items and no missing scores."""
    complete = loose[loose.n_items.eq(10)].dropna(subset=COMPLETE_SCORE_COLUMNS).copy()
    complete["depressed"] = (complete.cesd_raw >= 10).astype(int)
    complete = complete.drop(columns=["somatic_z", "depraffect_z", "posaffect_z"])
    complete = add_factor_z_scores(complete)
    return complete[OUTPUT_COLUMNS]


def score_ifls5() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_stata_df(
        RAW_IFLS_EXTRACTED / "IFLS5/hh14/b3b_kp.dta", convert_categoricals=False
    )
    df = raw.copy()
    df = score_item_frame(df, "IFLS5")
    agg = (
        df.groupby("pidlink")
        .agg(n_items=("score", "size"), cesd_raw=("score", "sum"))
        .reset_index()
    )
    agg["wave"] = "IFLS5"
    agg["cesd10_count"] = float("nan")
    df["wave"] = "IFLS5"
    return agg, df


def score_ifls4() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_stata_df(
        RAW_IFLS_EXTRACTED / "IFLS4/hh07/b3b_kp.dta", convert_categoricals=False
    )
    df = raw.copy()
    yes = df[df.kp01.isin([1, 3])].copy()
    yes["yes"] = (yes.kp01 == 1).astype(int)
    yes.loc[yes.kptype.isin(REVERSE_ITEMS), "yes"] = (
        1 - yes.loc[yes.kptype.isin(REVERSE_ITEMS), "yes"]
    )

    scored = score_item_frame(df, "IFLS4")
    agg = (
        yes.groupby("pidlink")
        .agg(
            n_items=("yes", "size"),
        )
        .reset_index()
    )
    agg = agg.merge(
        yes.groupby("pidlink").agg(cesd10_count=("yes", "sum")).reset_index(),
        on="pidlink",
        how="left",
        validate="1:1",
    ).merge(
        scored.groupby("pidlink").agg(cesd_raw=("score", "sum")).reset_index(),
        on="pidlink",
        how="left",
        validate="1:1",
    )
    agg["wave"] = "IFLS4"
    scored["wave"] = "IFLS4"
    return agg, scored


def main() -> None:
    scored4, items4 = score_ifls4()
    scored5, items5 = score_ifls5()
    loose = build_output([scored4, scored5], [items4, items5])
    loose.to_parquet(GENERATED_DATA / "24_cesd_scores_loose.parquet", index=False)
    log(
        f"wrote {len(loose):,} rows to "
        f"{GENERATED_DATA / '24_cesd_scores_loose.parquet'}"
    )

    out = loose.pipe(complete_only).pipe(CESD_SCORES_SCHEMA.validate)
    out.to_parquet(GENERATED_DATA / "24_cesd_scores.parquet", index=False)
    log(f"wrote {len(out):,} rows to {GENERATED_DATA / '24_cesd_scores.parquet'}")
    log(
        out.groupby("wave")
        .agg(
            n=("pidlink", "size"),
            n_items_med=("n_items", "median"),
            cesd_raw_mean=("cesd_raw", "mean"),
            cesd_raw_p50=("cesd_raw", "median"),
            depressed_pct=("depressed", lambda x: 100 * x.mean()),
        )
        .round(2),
        "DEBUG",
    )


if __name__ == "__main__":
    main()
