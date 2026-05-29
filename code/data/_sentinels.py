"""IFLS sentinel-code → NaN conversion utilities.

IFLS uses standardised sentinel codes for "Don't know / Refused" (xxx8 family) and
"Missing / Skipped" (xxx9 family) across all numeric fields. The exact code depends
on the field's width:

  width   don't know     missing
  -----   -----------    -----------
  1       8              9         (rare; 8/9 can also be legit data e.g. ages)
  2       98             99
  3       998            999
  4       9998           9999
  5       99998          99999
  8       99999998       99999999    (monetary "Rp." 8-digit)
  9       999999998      999999999   (monetary 9-digit)
  10      9999999998     9999999999  (monetary 10-digit)

Usage in a data script::

    from _sentinels import clean_money, clean_year, clean_age, clean_categorical

    df["expenditure"] = clean_money(df["expenditure"])
    df["last_loss_year"] = clean_year(df["tk46dy"])
    df["age"] = clean_age(df["ar09"])
    df["marital_raw"] = clean_categorical(df["ar13"], digits=1)

Every helper:
  - Coerces input to numeric (errors="coerce")
  - Returns NaN for any value matching a sentinel for the declared field width
  - Leaves all other values unchanged
"""
import pandas as pd


# Per-width sentinel sets
_SENTINELS_BY_DIGITS: dict[int, set[float]] = {
    1:  {8, 9},
    2:  {98, 99},
    3:  {998, 999},
    4:  {9998, 9999},
    5:  {99998, 99999},
    8:  {99999998, 99999999},
    9:  {999999998, 999999999},
    10: {9999999998, 9999999999},
}

# Pooled supersets for common variable types
_MONEY_SENTINELS = (
    _SENTINELS_BY_DIGITS[5]
    | _SENTINELS_BY_DIGITS[8]
    | _SENTINELS_BY_DIGITS[9]
    | _SENTINELS_BY_DIGITS[10]
)
_YEAR_SENTINELS = _SENTINELS_BY_DIGITS[2] | _SENTINELS_BY_DIGITS[4]
_AGE_SENTINELS = _SENTINELS_BY_DIGITS[2] | _SENTINELS_BY_DIGITS[3]


def clean_categorical(s: pd.Series, digits: int = 2) -> pd.Series:
    """Mask IFLS sentinels for a categorical/ordinal field of the given width.

    digits = 1 is supported but use with care — 8/9 may be real category values
    (e.g. occupation codes where 9 = Social Services). Inspect distributions first.
    """
    s = pd.to_numeric(s, errors="coerce")
    sentinels = _SENTINELS_BY_DIGITS.get(digits, set())
    return s.where(~s.isin(sentinels))


def clean_money(s: pd.Series) -> pd.Series:
    """Mask all IFLS monetary sentinels (5-, 8-, 9-, 10-digit) as NaN."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(~s.isin(_MONEY_SENTINELS))


def clean_year(s: pd.Series) -> pd.Series:
    """Mask year sentinels (98/99 in 2-digit fields, 9998/9999 in 4-digit fields).

    Real IFLS years are post-1900 in 4-digit form or 7..15 in 2-digit form,
    so neither 98/99 nor 9998/9999 are ever real values.
    """
    s = pd.to_numeric(s, errors="coerce")
    return s.where(~s.isin(_YEAR_SENTINELS))


def clean_month(s: pd.Series) -> pd.Series:
    """Mask 98/99 month sentinels. Real months are 1..12."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(~s.isin({98, 99}))


def clean_age(s: pd.Series) -> pd.Series:
    """Mask 98/99/998/999 age sentinels. Keep 8 and 9 (legitimate children's ages)."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(~s.isin(_AGE_SENTINELS))


def clean_count(s: pd.Series, max_real: int = 50) -> pd.Series:
    """Mask sentinels for count fields (e.g. tk46c = job-loss count).

    Anything above `max_real` is treated as a sentinel. Default 50 covers
    plausible count ranges while masking 98/99/998/999/etc.
    """
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= 0) & (s <= max_real))


def audit(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Quick audit: for each numeric column, report counts of common sentinel values.

    Useful for catching unhandled sentinels. Returns a DataFrame with one row per
    flagged column.
    """
    if cols is None:
        cols = [c for c in df.columns if df[c].dtype.kind in "if"]
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        n_2  = int(s.isin({98, 99}).sum())
        n_3  = int(s.isin({998, 999}).sum())
        n_4  = int(s.isin({9998, 9999}).sum())
        n_m  = int(s.isin(_MONEY_SENTINELS).sum())
        if n_2 + n_3 + n_4 + n_m == 0:
            continue
        rows.append({
            "column":     c,
            "n_98_99":    n_2,
            "n_998_999":  n_3,
            "n_9998_9999": n_4,
            "n_money_sentinel": n_m,
        })
    return pd.DataFrame(rows)
