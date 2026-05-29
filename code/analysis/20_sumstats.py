"""Build the summary-statistics table (Table 1 of the paper).

Reports descriptive statistics for the pooled analysis sample (n = 59,944):
  Panel A: Outcome (CES-D)
  Panel B: Heat exposure (Tmean, Tmax, Tmin)
  Panel C: Economic stressors (job loss, palm shock, fuel shock, etc.)
  Panel D: Demographics

Columns: Mean, SD, p25, Median, p75, N
Also reports a final "by exposure" panel showing the unexposed vs exposed
mean of CES-D for each of the three headline stressors.

Outputs:
  output/tables/table_sumstats.tex
  output/tables/table_sumstats.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "data" / "generated"
TAB = PROJECT / "output" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

from _table_input import load_table_input


def load_data() -> pd.DataFrame:
    df = load_table_input()
    df = df.dropna(subset=[
        "cesd_raw", "tmean_c", "tmax_c", "tmin_c",
        "kabupaten_code", "month", "year", "wave",
        "age", "female", "edu_yrs", "married", "widowed",
        "job_loss_within_yr", "palm_farmer_hh", "transport_share",
        "interview_date",
    ])
    # Match the analysis-sample restriction (drop singleton kabs)
    counts = df.kabupaten_code.value_counts()
    df = df[df.kabupaten_code.isin(counts[counts > 1].index)].copy()

    return df


def fmt(x, decimals=2):
    if pd.isna(x): return ""
    if abs(x) >= 100_000: return f"{x:,.0f}"
    return f"{x:,.{decimals}f}"


def stats_row(s, label, decimals=2, sample=None):
    if sample is not None:
        s = s[sample]
    s = s.dropna()
    return [label, fmt(s.mean(), decimals), fmt(s.std(), decimals),
            f"{len(s):,}"]


def main() -> None:
    df = load_data()
    n_pooled = len(df)
    n_ifls4 = (df.wave == "IFLS4").sum()
    n_ifls5 = (df.wave == "IFLS5").sum()
    print(f"Pooled sample: n = {n_pooled:,}  (IFLS4: {n_ifls4:,}; IFLS5: {n_ifls5:,})")

    # Build rows for each panel
    panels = [
        ("A. Mental-health outcome", [
            ("CES-D total score (0--30)",       "cesd_raw",    1),
            ("CES-D z-score (within wave)",     "cesd_z",      2),
            ("Depressed (CES-D $\\geq 10$)",      "depressed",   3),
        ]),
        ("B. Daily temperature exposure ($^{\\circ}$C)", [
            ("Daily mean temperature",          "tmean_c",     2),
            ("Daily maximum temperature",       "tmax_c",      2),
            ("Daily minimum temperature",       "tmin_c",      2),
        ]),
        ("C. Economic stressors", [
            ("Job loss within 12 months",       "job_loss_within_yr",      3),
            ("Palm-farmer household (any adult)", "palm_farmer_hh", 3),
            ("3-month palm-price decline",      "palm_3mo_decline",        3),
            ("Palm shock (PalmFarmerHH $\\times$ decline)", "palm_shock",      3),
            ("Post Nov-2014 fuel subsidy cut",  "post_subsidy",            3),
            ("Transport-spending share",        "transport_share",         3),
            ("Fuel shock (Post $\\times$ TransportShare)", "fuel_shock",    3),
        ]),
        ("D. Demographics", [
            ("Age (years)",                     "age",                     1),
            ("Female",                          "female",                  3),
            ("Years of schooling",              "edu_yrs",                 1),
            ("Married",                         "married",                 3),
            ("Widowed",                         "widowed",                 3),
            ("Per-capita expenditure (IDR/mo, 000s)", "pce_000",           0),
        ]),
    ]
    df["pce_000"] = df.pce / 1000.0  # express PCE in thousands of IDR for readability

    # --- Build LaTeX ---
    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Summary statistics}")
    lines.append(r"\label{tab:sumstats}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\begin{tabular}{l*{3}{r}}")
    lines.append(r"\toprule")
    lines.append(r" & Mean & SD & $N$ \\")
    lines.append(r"\midrule")
    for title, rows in panels:
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{title}}}}} \\")
        lines.append(r"\addlinespace[2pt]")
        for label, var, dec in rows:
            if var not in df.columns:
                continue
            row = stats_row(df[var], "\\quad " + label, decimals=dec)
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace[4pt]")

    lines.append(r"\midrule")
    lines.append(rf"Observations &  &  & {n_pooled:,} \\")
    lines.append(rf"\quad IFLS-4 (2007--2008) &  &  & {n_ifls4:,} \\")
    lines.append(rf"\quad IFLS-5 (2014--2015) &  &  & {n_ifls5:,} \\")
    lines.append(r"\quad Kabupaten clusters &  &  & " + f"{df.kabupaten_code.nunique():,}" + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    lines.append(r"\begin{tablenotes}[flushleft]")
    lines.append(r"\footnotesize")
    lines.append(
        r"\item \textit{Notes:} Summary statistics for the pooled analysis "
        r"sample of adults aged 15+ in IFLS-4 (2007--2008) and IFLS-5 (2014--2015), "
        r"after dropping singleton-kabupaten observations. Mental-health outcome "
        r"is the 10-item Andresen CES-D scored on the standard 0--3 frequency "
        r"scale and summed to 0--30; the z-score is computed within wave. "
        r"Daily temperature variables are ERA5-Land kabupaten polygon means on "
        r"the interview date. The job-loss indicator equals 1 if the respondent's "
        r"most recent job termination was within 365 days of interview. "
        r"PalmFarmer equals 1 for agricultural workers in palm-producing provinces; "
        r"the 3-month palm-price decline is the cumulative percentage drop in the "
        r"World Bank monthly palm-oil price over the three months preceding "
        r"interview, set to zero in non-decline months. The fuel-subsidy "
        r"indicator equals 1 for interviews on or after 18 November 2014. "
        r"Per-capita expenditure is nominal monthly household consumption divided "
        r"by household size, in thousands of Indonesian rupiah."
    )
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines) + "\n"
    (TAB / "table_sumstats.tex").write_text(tex, encoding="utf-8")
    print(f"\nwrote {TAB / 'table_sumstats.tex'}")

    # Machine-readable copy
    rows_csv = []
    for title, rows in panels:
        for label, var, dec in rows:
            if var not in df.columns:
                continue
            s = df[var].dropna()
            rows_csv.append({
                "panel": title, "label": label, "var": var,
                "mean": s.mean(), "sd": s.std(),
                "p25": s.quantile(0.25), "p50": s.median(), "p75": s.quantile(0.75),
                "min": s.min(), "max": s.max(), "n": len(s),
            })
    pd.DataFrame(rows_csv).to_csv(TAB / "table_sumstats.csv", index=False)
    print(f"wrote {TAB / 'table_sumstats.csv'}")

    # Console preview
    print("\n=== Console preview ===")
    print(f"{'':<45} {'Mean':>10} {'SD':>10} {'p25':>10} {'p50':>10} {'p75':>10} {'N':>10}")
    for title, rows in panels:
        print(f"\n{title}")
        for label, var, dec in rows:
            if var not in df.columns: continue
            s = df[var].dropna()
            print(f"  {label:<43} {s.mean():>10.{dec}f} {s.std():>10.{dec}f} "
                  f"{s.quantile(0.25):>10.{dec}f} {s.median():>10.{dec}f} "
                  f"{s.quantile(0.75):>10.{dec}f} {len(s):>10,}")


if __name__ == "__main__":
    main()
