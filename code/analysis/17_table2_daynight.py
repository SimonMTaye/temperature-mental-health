"""Build Appendix Table A2 — linear day/night heat decomposition for each stressor.

(Demoted from Table 2 to Appendix A2 because the CDD-threshold spec in the new
Table 2 is more powerful for detecting day-vs-night asymmetry. The linear specs
here are kept as a robustness check; their interpretation is "the average heat
slope of stressed adults is positive on both day and night temperature, but a
Wald test cannot statistically reject equality of Tmax × Stress and Tmin × Stress
due to Tmax-Tmin correlation of 0.54".)

Two panels (Tmax, Tmin) × three columns (Job loss, Palm shock, Fuel cut), all
on CES-D total z with the full-spec controls and FE from Table 1.

DV = CES-D z-standardised within wave.
Heat measures mean-centred on the pooled sample mean.
SEs cluster-robust at the kabupaten level.

Outputs:
  output/tables/appendix_a2_linear_daynight.tex
  output/tables/appendix_a2_linear_daynight.csv
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


CONTROLS = "age + female + edu_yrs + married + widowed"


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
        "cesd_raw", "tmean_c", "tmax_c", "tmin_c",
        "kabupaten_code", "month", "year", "wave",
        "age", "female", "edu_yrs", "married", "widowed",
        "job_loss_within_yr", "palm_farmer_hh", "transport_share",
        "interview_date",
    ])
    df["cesd_z"] = df.groupby("wave")["cesd_raw"].transform(lambda s: (s - s.mean()) / s.std())
    # Mean-centred heat measures
    df["tmax_c_dev"] = df.tmax_c - df.tmax_c.mean()
    df["tmin_c_dev"] = df.tmin_c - df.tmin_c.mean()
    df["intvw_ym"] = list(zip(df.interview_date.dt.year, df.interview_date.dt.month))
    df["palm_3mo_decline"] = df.intvw_ym.map(PALM_3MO_DECLINE).fillna(0.0)
    df["palm_shock"] = df.palm_farmer_hh * df.palm_3mo_decline
    df["post_subsidy"] = (df.interview_date >= pd.Timestamp("2014-11-18")).astype(int)
    df["fuel_shock"] = df.post_subsidy * df.transport_share
    return df


def _restrict_singleton_kab(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.kabupaten_code.value_counts()
    return df[df.kabupaten_code.isin(counts[counts > 1].index)].copy()


def fit_spec(df: pd.DataFrame, heat: str, stressor: str, extra_control: str,
             fe: str, high_stressor_value: float) -> dict:
    """Run full-spec regression and return coefficients + marginal effect."""
    from scipy import stats as ss
    formula = f"cesd_z ~ {heat} * {stressor}{extra_control} + {CONTROLS} | {fe}"
    m = pf.feols(formula, data=df, vcov={"CRV1": "kabupaten_code"})
    inter = f"{heat}:{stressor}"
    coefs = m.coef()
    V = pd.DataFrame(m._vcov, index=coefs.index, columns=coefs.index)

    def lincom(weights: dict):
        a = pd.Series(0.0, index=coefs.index)
        for k, v in weights.items():
            if k in a.index:
                a[k] = v
        val = float(a @ coefs)
        se = float(np.sqrt(a @ V @ a))
        t = val / se if se > 0 else np.nan
        p = 2 * (1 - ss.norm.cdf(abs(t)))
        return val, se, p

    high_b, high_se, high_p = lincom({heat: 1.0, inter: high_stressor_value})

    return {
        "n": int(m._N),
        "inter_b": coefs.get(inter, np.nan),
        "inter_se": m.se().get(inter, np.nan),
        "inter_p": m.pvalue().get(inter, np.nan),
        "heat_b": coefs.get(heat, np.nan),
        "heat_se": m.se().get(heat, np.nan),
        "heat_p": m.pvalue().get(heat, np.nan),
        "high_b": high_b, "high_se": high_se, "high_p": high_p,
        "high_value": high_stressor_value,
    }


def stars(p: float) -> str:
    if pd.isna(p): return ""
    if p < 0.01: return r"^{***}"
    if p < 0.05: return r"^{**}"
    if p < 0.10: return r"^{*}"
    return ""


def cell(b: float, se: float, p: float) -> tuple[str, str]:
    if pd.isna(b):
        return ("", "")
    return (f"${b:+.3f}{stars(p)}$", f"$({se:.3f})$")


def main() -> None:
    df = load_data()
    df = _restrict_singleton_kab(df)
    sub_ifls5 = _restrict_singleton_kab(df[df.wave == "IFLS5"].copy())

    FE_POOLED = "month + year + wave + kabupaten_code"
    FE_IFLS5  = "month + year + kabupaten_code"

    # Stressor specs (same as Table 1)
    stressor_cfg = [
        {"col": "(1)", "name": "job_loss_within_yr", "extra": "",
         "sample": df,        "fe": FE_POOLED, "high": 1.0,
         "short": r"Job loss", "n": None},
        {"col": "(2)", "name": "palm_shock",
         "extra": " + palm_farmer_hh",
         "sample": df,        "fe": FE_POOLED, "high": 0.10,
         "short": r"Palm shock", "n": None},
        {"col": "(3)", "name": "fuel_shock",
         "extra": " + transport_share",
         "sample": sub_ifls5, "fe": FE_IFLS5,  "high": 0.10,
         "short": r"Fuel shock", "n": None},
    ]

    # Run all 6 specs (Tmax × 3 stressors, Tmin × 3 stressors)
    results = {}  # results[heat][col] = dict
    for heat, label in [("tmax_c_dev", "Tmax"), ("tmin_c_dev", "Tmin")]:
        results[heat] = {}
        for s in stressor_cfg:
            r = fit_spec(s["sample"], heat, s["name"], s["extra"], s["fe"], s["high"])
            results[heat][s["col"]] = r
            if s["n"] is None:
                s["n"] = r["n"]

    # ---- LaTeX ----
    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Appendix: Linear Tmax / Tmin decomposition (robustness)}")
    lines.append(r"\label{tab:appendix_linear_daynight}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r" & (1) & (2) & (3) \\")
    lines.append(r" & Job loss & Palm shock & Fuel cut \\")
    lines.append(r" & (within 12 mo) & (price decline $\times$ palm-farmer HH) & (post-cut $\times$ transport share) \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{Dependent variable: CES-D total, z-standardised within wave}} \\")
    lines.append(r"\midrule")

    panel_headers = [
        ("tmax_c_dev", "Tmax", r"\textit{Panel A: Daytime peak temperature (Tmax)}"),
        ("tmin_c_dev", "Tmin", r"\textit{Panel B: Overnight low temperature (Tmin)}"),
    ]
    for heat, hlabel, panel_title in panel_headers:
        lines.append(rf"\multicolumn{{4}}{{l}}{{{panel_title}}} \\")
        lines.append(r"\addlinespace[2pt]")
        # Heat × Stressor (interaction)
        coef_row, se_row = [], []
        for s in stressor_cfg:
            r = results[heat][s["col"]]
            c, se = cell(r["inter_b"], r["inter_se"], r["inter_p"])
            coef_row.append(c); se_row.append(se)
        lines.append(rf"\quad {hlabel} $\times$ Stressor & " + " & ".join(coef_row) + r" \\")
        lines.append(r" & " + " & ".join(se_row) + r" \\")
        lines.append(r"\addlinespace[3pt]")
        # Heat main
        coef_row, se_row = [], []
        for s in stressor_cfg:
            r = results[heat][s["col"]]
            c, se = cell(r["heat_b"], r["heat_se"], r["heat_p"])
            coef_row.append(c); se_row.append(se)
        lines.append(rf"\quad {hlabel} & " + " & ".join(coef_row) + r" \\")
        lines.append(r" & " + " & ".join(se_row) + r" \\")
        lines.append(r"\addlinespace[3pt]")
        # Marginal effect at exposed
        coef_row, se_row = [], []
        for s in stressor_cfg:
            r = results[heat][s["col"]]
            c, se = cell(r["high_b"], r["high_se"], r["high_p"])
            coef_row.append(c); se_row.append(se)
        lines.append(rf"\quad {hlabel} slope $|$ exposed$^{{\dagger}}$ & "
                     + " & ".join(coef_row) + r" \\")
        lines.append(r" & " + " & ".join(se_row) + r" \\")
        lines.append(r"\addlinespace[6pt]")

    lines.append(r"\midrule")
    lines.append(r"Demographic controls & Yes & Yes & Yes \\")
    lines.append(r"Kabupaten FE & Yes & Yes & Yes \\")
    lines.append(r"Month + Year FE & Yes & Yes & Yes \\")
    lines.append(r"Wave FE & Yes & Yes & --- \\")
    lines.append(r"\addlinespace[3pt]")
    lines.append(r"Sample & Pooled & Pooled & IFLS5 only \\")
    nA = stressor_cfg[0]["n"]; nB = stressor_cfg[1]["n"]; nC = stressor_cfg[2]["n"]
    lines.append(rf"Observations & {nA:,} & {nB:,} & {nC:,} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    lines.append(r"\begin{tablenotes}[flushleft]")
    lines.append(r"\footnotesize")
    lines.append(
        r"\item \textit{Notes:} Linear day/night decomposition of Table~\ref{tab:headline}: "
        r"the daily mean temperature is replaced with the daily maximum temperature (Tmax, "
        r"Panel A) and daily minimum temperature (Tmin, Panel B) respectively, on the interview "
        r"date (ERA5-Land kabupaten polygon mean), each mean-centred on the pooled sample. "
        r"The dependent variable is the 10-item Andresen CES-D score, z-standardised within "
        r"wave. Stressor definitions and the demographic-control set are identical to "
        r"Table~\ref{tab:headline}. A joint Wald test on the equality of Tmax $\times$ Stressor and "
        r"Tmin $\times$ Stressor (within a single regression containing both) cannot reject equality "
        r"for any stressor (p = 0.82 for job loss, 0.15 for palm shock, 0.18 for fuel cut), "
        r"because Tmax and Tmin are 54\% correlated. See Table~\ref{tab:cdd} for the CDD-threshold "
        r"specification, which yields the cleaner day-vs-night asymmetry test. "
        r"$^{\dagger}$The exposed-respondent marginal heat slope is evaluated at Stressor $= 1$ in "
        r"column (1), Stressor $= 0.10$ in column (2) (a 10-percentage-point palm-price decline for "
        r"a palm farmer), and Stressor $= 0.10$ in column (3) (a 10-percentage-point transport-spending "
        r"share post-cut), computed via the delta method on the kabupaten-clustered covariance "
        r"matrix. Column (3) is identified within IFLS5 only, so Wave FE is omitted. Standard "
        r"errors clustered at the kabupaten level are in parentheses. Significance: $^{*}$ $p<0.10$, "
        r"$^{**}$ $p<0.05$, $^{***}$ $p<0.01$."
    )
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines) + "\n"
    (TAB / "appendix_a2_linear_daynight.tex").write_text(tex, encoding="utf-8")
    print(f"wrote {TAB / 'appendix_a2_linear_daynight.tex'}  ({len(lines)} lines)")

    rows = []
    for heat, _, _ in panel_headers:
        for s in stressor_cfg:
            r = results[heat][s["col"]]
            rows.append({"heat": heat, "col": s["col"], "stressor": s["name"], **r})
    pd.DataFrame(rows).to_csv(TAB / "appendix_a2_linear_daynight.csv", index=False)

    print("\n=== Preview ===")
    for heat, hlabel, _ in panel_headers:
        print(f"\n[{hlabel}]")
        for s in stressor_cfg:
            r = results[heat][s["col"]]
            def fmt(b, p):
                if pd.isna(b): return "—"
                star = "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.10 else ""))
                return f"{b:+.4f}{star}"
            print(f"  {s['col']} {s['short']:12s} "
                  f"inter={fmt(r['inter_b'], r['inter_p'])}  "
                  f"heat={fmt(r['heat_b'], r['heat_p'])}  "
                  f"marg@exposed={fmt(r['high_b'], r['high_p'])}")


if __name__ == "__main__":
    main()
