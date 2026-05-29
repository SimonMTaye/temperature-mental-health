"""Build Table 1 — the headline interaction table on CES-D total.

One column per stressor (Job loss, Palm shock, Fuel cut × transport-share), each
showing the full specification with all fixed effects and demographic controls.

DV = CES-D z-standardised within wave. Heat = mean-centred Tmean.
SEs cluster-robust at the kabupaten level throughout.

Outputs:
  output/tables/table1_headline.tex
  output/tables/table1_headline.csv  (machine-readable copy)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "data" / "generated"
TAB = PROJECT / "output" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT / "code" / "analysis"))
from importlib import import_module
mod14 = import_module("14_unified_refined")
PALM_3MO_DECLINE = mod14.PALM_3MO_DECLINE


# ---------------- data ---------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(OUT / "analysis_dataset.parquet")
    fin  = pd.read_parquet(OUT / "financial_shocks.parquet")
    fin2 = pd.read_parquet(OUT / "financial_shocks_v2.parquet")
    df = df.merge(fin[["pidlink", "wave", "job_loss_within_yr"]],
                  on=["pidlink", "wave"], how="left")
    df = df.merge(fin2[["pidlink", "wave", "palm_farmer_hh", "transport_share"]],
                  on=["pidlink", "wave"], how="left")
    df["female"] = (df.sex == "F").astype(int)

    df = df.dropna(subset=[
        "cesd_raw", "tmean_c", "kabupaten_code", "month", "year", "wave",
        "age", "female", "edu_yrs", "married", "widowed",
        "job_loss_within_yr", "palm_farmer_hh", "transport_share",
        "interview_date",
    ])

    df["cesd_z"] = df.groupby("wave")["cesd_raw"].transform(lambda s: (s - s.mean()) / s.std())
    df["heat_c_dev"] = df.tmean_c - df.tmean_c.mean()

    df["intvw_ym"] = list(zip(df.interview_date.dt.year, df.interview_date.dt.month))
    df["palm_3mo_decline"] = df.intvw_ym.map(PALM_3MO_DECLINE).fillna(0.0)
    df["palm_shock"] = df.palm_farmer_hh * df.palm_3mo_decline

    df["post_subsidy"] = (df.interview_date >= pd.Timestamp("2014-11-18")).astype(int)
    df["fuel_shock"] = df.post_subsidy * df.transport_share
    return df


# ---------------- spec grid ---------------------------------------------------------

CONTROLS = "age + female + edu_yrs + married + widowed"


def fit_pooled_heat(df: pd.DataFrame, fe: str) -> dict:
    """Pooled regression with no stressor and no interaction.

    Estimates the unconditional heat slope under the same FE and demographic
    controls as the interacted columns. Used for the literature-reconciliation
    point: heat alone has a near-zero effect in the average sample, but the
    Heat x Stressor interactions in the other columns are positive and large.
    """
    formula = f"cesd_z ~ heat_c_dev + {CONTROLS} | {fe}"
    m = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_code"})
    return {
        "n":       int(m._N),
        "heat_b":  float(m.coef()["heat_c_dev"]),
        "heat_se": float(m.se()["heat_c_dev"]),
        "heat_p":  float(m.pvalue()["heat_c_dev"]),
    }


def fit_full_spec(df: pd.DataFrame, stressor: str, extra_control: str, fe: str,
                  high_stressor_value: float = 1.0) -> dict:
    """Run the full-spec regression and return raw regression coefficients plus
    heat-slope marginal effects at low (= 0) and high (= high_stressor_value) stressor.

    The "high" value is 1 for binary stressors (job loss) and 0.10 for continuous
    stressors interpreted "per 10 percentage points" (palm_shock, fuel_shock)."""
    formula = (
        f"cesd_z ~ heat_c_dev * {stressor}{extra_control} + {CONTROLS} | {fe}"
    )
    m = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_code"})
    inter = f"heat_c_dev:{stressor}"

    # lincom: heat slope at stressor=v  ==> β_heat + v · β_interaction
    from scipy import stats as scipy_stats
    coefs = m.coef()
    V = pd.DataFrame(m._vcov, index=coefs.index, columns=coefs.index)

    def lincom(weights: dict) -> tuple[float, float, float]:
        a = pd.Series(0.0, index=coefs.index)
        for k, v in weights.items():
            if k in a.index:
                a[k] = v
        val = float(a @ coefs)
        se = float(np.sqrt(a @ V @ a))
        t = val / se if se > 0 else np.nan
        p = 2 * (1 - scipy_stats.norm.cdf(abs(t)))
        return val, se, p

    # Marginal effect of heat at "treated" (stressor=high)
    heat_high_b, heat_high_se, heat_high_p = lincom({"heat_c_dev": 1.0, inter: high_stressor_value})

    return {
        "n": int(m._N),
        # Raw regression coefficients
        "inter_b": coefs.get(inter, np.nan),
        "inter_se": m.se().get(inter, np.nan),
        "inter_p": m.pvalue().get(inter, np.nan),
        "heat_b":   coefs.get("heat_c_dev", np.nan),
        "heat_se":  m.se().get("heat_c_dev", np.nan),
        "heat_p":   m.pvalue().get("heat_c_dev", np.nan),
        "stress_b":  coefs.get(stressor, np.nan),
        "stress_se": m.se().get(stressor, np.nan),
        "stress_p":  m.pvalue().get(stressor, np.nan),
        # Marginal effect at "treated" value
        "heat_high_b": heat_high_b, "heat_high_se": heat_high_se, "heat_high_p": heat_high_p,
        "high_value": high_stressor_value,
    }


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01: return r"^{***}"
    if p < 0.05: return r"^{**}"
    if p < 0.10: return r"^{*}"
    return ""


def cell(b: float, se: float, p: float) -> tuple[str, str]:
    """Return (coefficient row, SE row) LaTeX cells."""
    if pd.isna(b):
        return ("", "")
    return (f"${b:+.3f}{stars(p)}$", f"$({se:.3f})$")


# ---------------- run all panels ----------------------------------------------------

def _restrict_singleton_kab(df: pd.DataFrame) -> pd.DataFrame:
    """Drop singleton kabupatens so the FE spec doesn't lose observations."""
    counts = df.kabupaten_code.value_counts()
    return df[df.kabupaten_code.isin(counts[counts > 1].index)].copy()


def main() -> None:
    df = load_data()
    df = _restrict_singleton_kab(df)
    print(f"Pooled sample (after singleton-kab restriction): {len(df):,}")

    sub_ifls5 = df[df.wave == "IFLS5"].copy()
    sub_ifls5 = _restrict_singleton_kab(sub_ifls5)
    print(f"IFLS5-only sub-sample: {len(sub_ifls5):,}")

    # Full spec for each stressor: kab + month + year + wave FE + demographic controls
    FE_POOLED = "month + year + wave + kabupaten_code"
    FE_IFLS5  = "month + year + kabupaten_code"          # wave is constant in IFLS5

    # Column (1): pooled regression of CES-D z on heat, no stressor, no interaction.
    pooled = fit_pooled_heat(df, FE_POOLED)

    panels = [
        {"col": "(2)",  "label": "Job loss within 12 months",
         "header": r"Heat $\times$ Job loss",
         "stress_unit": "binary, 0/1",
         "result": fit_full_spec(df, "job_loss_within_yr",
                                  extra_control="", fe=FE_POOLED,
                                  high_stressor_value=1.0),
         "sample": "Pooled IFLS4+5",
        },
        {"col": "(3)",  "label": r"Palm-price shock $\times$ palm farmer",
         "header": r"Heat $\times$ Palm shock",
         "stress_unit": "per 10 pp 3-mo decline",
         "result": fit_full_spec(df, "palm_shock",
                                  extra_control=" + palm_farmer_hh",
                                  fe=FE_POOLED,
                                  high_stressor_value=0.10),
         "sample": "Pooled IFLS4+5",
        },
        {"col": "(4)",  "label": r"Fuel-subsidy cut $\times$ transport share",
         "header": r"Heat $\times$ Fuel shock",
         "stress_unit": "per 10 pp transport share",
         "result": fit_full_spec(sub_ifls5, "fuel_shock",
                                  extra_control=" + transport_share",
                                  fe=FE_IFLS5,
                                  high_stressor_value=0.10),
         "sample": "IFLS5 only",
        },
    ]

    # --- Build LaTeX body (tabular only) ---
    body = []
    body.append(r"\begin{tabular}{lcccc}")
    body.append(r"\toprule")
    body.append(r" & (1) & (2) & (3) & (4) \\")
    body.append(r" & Pooled & Job loss & Palm shock & Fuel cut \\")
    body.append(r" & (no interaction) & (within 12 mo) & (price decline $\times$ palm-farmer HH) & (post-cut $\times$ transport share) \\")
    body.append(r"\midrule")
    body.append(r"\multicolumn{5}{l}{\textit{Dependent variable: CES-D total, z-standardised within wave}} \\")
    body.append(r"\midrule")

    # --- Section A: Raw regression coefficients ---
    body.append(r"\multicolumn{5}{l}{\textit{A. Regression coefficients}} \\")
    body.append(r"\addlinespace[2pt]")

    DASH_C, DASH_SE = "---", ""

    # Heat × Stress interaction (no entry in column 1)
    coef_row, se_row = [DASH_C], [DASH_SE]
    for p in panels:
        r = p["result"]
        c, s = cell(r["inter_b"], r["inter_se"], r["inter_p"])
        coef_row.append(c); se_row.append(s)
    body.append(r"\quad Heat $\times$ Stressor & " + " & ".join(coef_row) + r" \\")
    body.append(r" & " + " & ".join(se_row) + r" \\")
    body.append(r"\addlinespace[3pt]")

    # Heat (column 1 = unconditional slope; columns 2-4 = slope at Stressor=0)
    pooled_c, pooled_s = cell(pooled["heat_b"], pooled["heat_se"], pooled["heat_p"])
    coef_row, se_row = [pooled_c], [pooled_s]
    for p in panels:
        r = p["result"]
        c, s = cell(r["heat_b"], r["heat_se"], r["heat_p"])
        coef_row.append(c); se_row.append(s)
    body.append(r"\quad Heat & " + " & ".join(coef_row) + r" \\")
    body.append(r" & " + " & ".join(se_row) + r" \\")
    body.append(r"\addlinespace[3pt]")

    # Stressor (main effect): no entry in column 1
    coef_row, se_row = [DASH_C], [DASH_SE]
    for p in panels:
        r = p["result"]
        c, s = cell(r["stress_b"], r["stress_se"], r["stress_p"])
        coef_row.append(c); se_row.append(s)
    body.append(r"\quad Stressor & " + " & ".join(coef_row) + r" \\")
    body.append(r" & " + " & ".join(se_row) + r" \\")
    body.append(r"\addlinespace[6pt]")

    # --- Section B: Marginal effect of heat at "treated" ---
    body.append(r"\multicolumn{5}{l}{\textit{B. Marginal effect of heat at exposed reference value$^{\dagger}$}} \\")
    body.append(r"\addlinespace[2pt]")
    coef_row, se_row = [DASH_C], [DASH_SE]
    for p in panels:
        r = p["result"]
        c, s = cell(r["heat_high_b"], r["heat_high_se"], r["heat_high_p"])
        coef_row.append(c); se_row.append(s)
    body.append(r"\quad Heat slope $|$ exposed$^{\dagger}$ & " + " & ".join(coef_row) + r" \\")
    body.append(r" & " + " & ".join(se_row) + r" \\")
    body.append(r"\midrule")

    # FE / controls indicators (column 4 = IFLS5 only, so Wave FE n/a)
    yes = "Yes"
    fe_rows = [
        ("Demographic controls",           [yes, yes, yes, yes]),
        ("Kabupaten FE",                   [yes, yes, yes, yes]),
        ("Month + Year FE",                [yes, yes, yes, yes]),
        ("Wave FE",                        [yes, yes, yes, "---"]),
    ]
    for label, vals in fe_rows:
        body.append(label + " & " + " & ".join(vals) + r" \\")
    body.append(r"\addlinespace[3pt]")
    body.append(r"Sample & Pooled & Pooled & Pooled & IFLS5 only \\")
    body.append(
        r"Observations & "
        + f"{pooled['n']:,} & "
        + f"{panels[0]['result']['n']:,} & "
        + f"{panels[1]['result']['n']:,} & "
        + f"{panels[2]['result']['n']:,} \\\\"
    )
    body.append(r"\bottomrule")
    body.append(r"\end{tabular}")

    body_tex = "\n".join(body) + "\n"
    (TAB / "table1_headline_body.tex").write_text(body_tex, encoding="utf-8")
    print(f"\nwrote {TAB / 'table1_headline_body.tex'}")

    # --- Build full LaTeX (with caption, label, threeparttable, notes) ---
    notes = (
        r"\item \textit{Notes:} The dependent variable is the 10-item Andresen CES-D score, "
        r"z-standardised within wave. Heat is the daily mean temperature on the interview date "
        r"(ERA5-Land kabupaten polygon mean), centred on the pooled sample mean of 24.83$^\circ$C. "
        r"Column (1) reports the unconditional heat slope from a regression with no stressor and no "
        r"interaction term. Columns (2)--(4) add a stressor indicator and its interaction with heat; "
        r"in those columns the ``Heat'' row reports the slope at Stressor $= 0$. "
        r"The job-loss indicator equals 1 if the respondent's most recent job termination occurred "
        r"within 365 days of the interview. The palm shock equals the cumulative percentage drop in "
        r"the World Bank monthly palm-oil price over the 3 months preceding interview, interacted "
        r"with a household-level indicator equal to one if any adult in the household is an "
        r"agricultural worker in a palm-producing province (capturing within-household income "
        r"spillovers from the palm-farming member); this household-level palm-farmer indicator is "
        r"included as a separate control. The fuel shock equals the household's monthly "
        r"transport-spending share interacted with an indicator for interviews on or after "
        r"18 November 2014 (Indonesia's fuel-subsidy cut); transport share is included as a "
        r"separate control. Demographic controls are age, an indicator for female, years of "
        r"completed schooling, and indicators for being married and being widowed. "
        r"$^{\dagger}$The ``exposed'' heat slope is evaluated at Stressor $= 1$ in column (2) "
        r"(recent job loss) and at Stressor $= 0.10$ in columns (3)--(4) (a 10-percentage-point "
        r"palm-price decline for a household with at least one palm-farmer adult in column (3); "
        r"a 10-percentage-point transport-spending share post-cut in column (4)). Marginal effects "
        r"are computed via the delta method on the kabupaten-clustered covariance matrix. "
        r"Column (4) is identified within IFLS5 only, so Wave FE is omitted (mechanically collinear "
        r"with the constant). Standard errors clustered at the kabupaten level are in parentheses. "
        r"Significance: $^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$."
    )

    full = []
    full.append(r"\begin{table}[!htbp]")
    full.append(r"\centering")
    full.append(r"\caption{Heat amplifies the mental-health cost of acute economic stress}")
    full.append(r"\label{tab:headline}")
    full.append(r"\begin{threeparttable}")
    full.extend(body)
    full.append(r"\begin{tablenotes}[flushleft]")
    full.append(r"\footnotesize")
    full.append(notes)
    full.append(r"\end{tablenotes}")
    full.append(r"\end{threeparttable}")
    full.append(r"\end{table}")

    full_tex = "\n".join(full) + "\n"
    (TAB / "table1_headline.tex").write_text(full_tex, encoding="utf-8")
    print(f"wrote {TAB / 'table1_headline.tex'}")

    # Machine-readable backup
    rows = []
    rows.append({"col": "(1)", "label": "Pooled (no interaction)", "sample": "Pooled IFLS4+5",
                 **{f"pooled_{k}": v for k, v in pooled.items()}})
    for p in panels:
        rows.append({"col": p["col"], "label": p["label"], "sample": p["sample"], **p["result"]})
    pd.DataFrame(rows).to_csv(TAB / "table1_headline.csv", index=False)
    print(f"wrote {TAB / 'table1_headline.csv'}")

    print("\n=== Preview ===")
    print(f"\n(1) Pooled (no interaction)  (n={pooled['n']:,})")
    print(f"  Heat (unconditional):  β = {pooled['heat_b']:+.4f}{stars(pooled['heat_p'])}  ({pooled['heat_se']:.4f})")
    for p in panels:
        r = p["result"]
        print(f"\n{p['col']} {p['label']}  (n={r['n']:,})")
        print(f"  Heat × Stress:         β = {r['inter_b']:+.4f}{stars(r['inter_p'])}  ({r['inter_se']:.4f})")
        print(f"  Heat (at S=0):         β = {r['heat_b']:+.4f}{stars(r['heat_p'])}  ({r['heat_se']:.4f})")
        print(f"  Stressor (main):       β = {r['stress_b']:+.4f}{stars(r['stress_p'])}  ({r['stress_se']:.4f})")
        print(f"  Marg eff exposed:      β = {r['heat_high_b']:+.4f}{stars(r['heat_high_p'])}  ({r['heat_high_se']:.4f})  [at stressor={r['high_value']}]")


if __name__ == "__main__":
    main()
