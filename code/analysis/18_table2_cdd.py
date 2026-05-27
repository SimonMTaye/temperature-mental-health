"""Build Table 2 — extreme-heat-threshold (CDD) decomposition of heat × stressor.

Two panels (Tmax extreme, Tmin warm-night) × three columns (Job loss, Palm shock,
Fuel cut). Each cell is the heat × stressor interaction from a separate full-spec
regression on CES-D total z.

Thresholds are chosen with literature justification:
  CDD Tmax > 30°C : outdoor-labor productivity decline (ILO 2019; Park et al. 2021).
  CDD Tmax > 32°C : NIOSH/ACGIH heat-stress moderate-work threshold; Mullins & White
                    (2019, AEJ:Applied) primary nonlinear Tmax spec.
  CDD Tmin > 23°C : "warm tropical night" — sleep onset/depth disrupted above this
                    in tropical-acclimated samples (Okamoto-Mizuno & Mizuno 2012).
  CDD Tmin > 24°C : WHO "tropical night" definition.

Sample shares of obs with CDD > 0:
  Tmax > 30: 17.2%   Tmax > 32: 4.4%
  Tmin > 23: 39.2%   Tmin > 24: 11.7%

Outputs:
  output/tables/table2_cdd.tex
  output/tables/table2_cdd.csv
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
        "kab_code", "month", "year", "wave",
        "age", "female", "edu_yrs", "married", "widowed",
        "job_loss_within_yr", "palm_farmer_hh", "transport_share",
        "interview_date",
    ])
    df["cesd_z"] = df.groupby("wave")["cesd_raw"].transform(lambda s: (s - s.mean()) / s.std())
    # Cooling-degree-day measures
    df["cdd_tmax30"] = (df.tmax_c - 30.0).clip(lower=0)
    df["cdd_tmax32"] = (df.tmax_c - 32.0).clip(lower=0)
    df["cdd_tmin23"] = (df.tmin_c - 23.0).clip(lower=0)
    df["cdd_tmin24"] = (df.tmin_c - 24.0).clip(lower=0)
    df["intvw_ym"] = list(zip(df.interview_date.dt.year, df.interview_date.dt.month))
    df["palm_3mo_decline"] = df.intvw_ym.map(PALM_3MO_DECLINE).fillna(0.0)
    df["palm_shock"] = df.palm_farmer_hh * df.palm_3mo_decline
    df["post_subsidy"] = (df.interview_date >= pd.Timestamp("2014-11-18")).astype(int)
    df["fuel_shock"] = df.post_subsidy * df.transport_share
    return df


def _restrict_singleton_kab(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.kab_code.value_counts()
    return df[df.kab_code.isin(counts[counts > 1].index)].copy()


def fit_cdd(df: pd.DataFrame, heat: str, stressor: str, extra_control: str, fe: str) -> dict:
    formula = f"cesd_z ~ {heat} * {stressor}{extra_control} + {CONTROLS} | {fe}"
    m = pf.feols(formula, data=df, vcov={"CRV1": "kab_code"})
    inter = f"{heat}:{stressor}"
    return {
        "n":         int(m._N),
        "inter_b":   m.coef().get(inter, np.nan),
        "inter_se":  m.se().get(inter, np.nan),
        "inter_p":   m.pvalue().get(inter, np.nan),
        "heat_b":    m.coef().get(heat, np.nan),
        "heat_se":   m.se().get(heat, np.nan),
        "heat_p":    m.pvalue().get(heat, np.nan),
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

    FE_POOLED = "month + year + wave + kab_code"
    FE_IFLS5  = "month + year + kab_code"

    stressors = [
        {"col": "(1)", "name": "job_loss_within_yr", "extra": "",
         "sample": df, "fe": FE_POOLED},
        {"col": "(2)", "name": "palm_shock",         "extra": " + palm_farmer_hh",
         "sample": df, "fe": FE_POOLED},
        {"col": "(3)", "name": "fuel_shock",         "extra": " + transport_share",
         "sample": sub_ifls5, "fe": FE_IFLS5},
    ]

    cdd_specs = [
        ("cdd_tmax30", r"CDD Tmax $> 30^{\circ}$C",  "Daytime (Tmax)"),
        ("cdd_tmax32", r"CDD Tmax $> 32^{\circ}$C",  "Daytime (Tmax)"),
        ("cdd_tmin23", r"CDD Tmin $> 23^{\circ}$C",  "Nighttime (Tmin)"),
        ("cdd_tmin24", r"CDD Tmin $> 24^{\circ}$C",  "Nighttime (Tmin)"),
    ]

    # results[heat][col] = dict
    results = {}
    for heat, _, _ in cdd_specs:
        results[heat] = {}
        for s in stressors:
            r = fit_cdd(s["sample"], heat, s["name"], s["extra"], s["fe"])
            results[heat][s["col"]] = r

    # --- LaTeX ---
    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Extreme-heat thresholds: heat $\times$ stressor decomposition by daytime peak and overnight low}")
    lines.append(r"\label{tab:cdd}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r" & (1) & (2) & (3) \\")
    lines.append(r" & Job loss & Palm shock & Fuel cut \\")
    lines.append(r" & (within 12 mo) & (price decline $\times$ palm-farmer HH) & (post-cut $\times$ transport share) \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{Dependent variable: CES-D total, z-standardised within wave}} \\")
    lines.append(r"\multicolumn{4}{l}{\textit{Each cell: heat-CDD $\times$ stressor coefficient from a separate full-spec regression}} \\")
    lines.append(r"\midrule")

    # Panel A: Daytime extreme heat (Tmax)
    lines.append(r"\multicolumn{4}{l}{\textit{Panel A: Daytime extreme heat (Tmax-based CDD)}} \\")
    lines.append(r"\addlinespace[2pt]")
    for heat, label, _ in cdd_specs[:2]:
        coef_row, se_row = [], []
        for s in stressors:
            r = results[heat][s["col"]]
            c, se = cell(r["inter_b"], r["inter_se"], r["inter_p"])
            coef_row.append(c); se_row.append(se)
        lines.append(rf"\quad {label} $\times$ Stressor & " + " & ".join(coef_row) + r" \\")
        lines.append(r" & " + " & ".join(se_row) + r" \\")
        lines.append(r"\addlinespace[3pt]")
    lines.append(r"\addlinespace[3pt]")

    # Panel B: Overnight warm nights (Tmin)
    lines.append(r"\multicolumn{4}{l}{\textit{Panel B: Warm nights (Tmin-based CDD)}} \\")
    lines.append(r"\addlinespace[2pt]")
    for heat, label, _ in cdd_specs[2:]:
        coef_row, se_row = [], []
        for s in stressors:
            r = results[heat][s["col"]]
            c, se = cell(r["inter_b"], r["inter_se"], r["inter_p"])
            coef_row.append(c); se_row.append(se)
        lines.append(rf"\quad {label} $\times$ Stressor & " + " & ".join(coef_row) + r" \\")
        lines.append(r" & " + " & ".join(se_row) + r" \\")
        lines.append(r"\addlinespace[3pt]")
    lines.append(r"\midrule")

    # FE / controls
    lines.append(r"Demographic controls & Yes & Yes & Yes \\")
    lines.append(r"Kabupaten FE & Yes & Yes & Yes \\")
    lines.append(r"Month + Year FE & Yes & Yes & Yes \\")
    lines.append(r"Wave FE & Yes & Yes & --- \\")
    lines.append(r"\addlinespace[3pt]")
    lines.append(r"Sample & Pooled & Pooled & IFLS5 only \\")
    nA = results["cdd_tmax30"]["(1)"]["n"]
    nB = results["cdd_tmax30"]["(2)"]["n"]
    nC = results["cdd_tmax30"]["(3)"]["n"]
    lines.append(rf"Observations & {nA:,} & {nB:,} & {nC:,} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    lines.append(r"\begin{tablenotes}[flushleft]")
    lines.append(r"\footnotesize")
    lines.append(
        r"\item \textit{Notes:} The dependent variable is the 10-item Andresen CES-D score, "
        r"z-standardised within wave. Each cell reports the interaction coefficient from a "
        r"separate full-specification regression of CES-D z on a heat-threshold cooling-degree-day "
        r"(CDD) measure interacted with the stressor, plus the same demographic controls and fixed "
        r"effects as Table~\ref{tab:headline}. CDD measures equal $\max(T - \tau, 0)$ for the given "
        r"threshold $\tau$, capturing the magnitude of heat exposure above the threshold on the "
        r"interview date. Threshold choices follow the heat-health and heat-labour literature: "
        r"$30^{\circ}$C Tmax is the outdoor-labour productivity threshold (ILO 2019; Park, Pankratz "
        r"and Behrer 2021); $32^{\circ}$C Tmax is the NIOSH/ACGIH heat-stress threshold for moderate-"
        r"intensity work and the primary Tmax spec in Mullins and White (2019); $23^{\circ}$C Tmin "
        r"is the sleep-disruption threshold in tropical-acclimated samples (Okamoto-Mizuno and "
        r"Mizuno 2012); $24^{\circ}$C Tmin is the WHO tropical-night definition. The share of "
        r"observations with CDD $> 0$ is $17.2\%$ (Tmax $> 30$), $4.4\%$ (Tmax $> 32$), $39.2\%$ "
        r"(Tmin $> 23$) and $11.7\%$ (Tmin $> 24$). Stressor definitions are identical to "
        r"Table~\ref{tab:headline}. Standard errors clustered at the kabupaten level are reported "
        r"in parentheses. Significance: $^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$."
    )
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines) + "\n"
    (TAB / "table2_cdd.tex").write_text(tex, encoding="utf-8")
    print(f"wrote {TAB / 'table2_cdd.tex'}  ({len(lines)} lines)")

    rows = []
    for heat, _, _ in cdd_specs:
        for s in stressors:
            r = results[heat][s["col"]]
            rows.append({"heat": heat, "col": s["col"], "stressor": s["name"], **r})
    pd.DataFrame(rows).to_csv(TAB / "table2_cdd.csv", index=False)

    print("\n=== Preview ===")
    for heat, label, panel in cdd_specs:
        print(f"\n[{panel}] {label}")
        for s in stressors:
            r = results[heat][s["col"]]
            def fmt(b, p):
                if pd.isna(b): return "—"
                star_s = "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.10 else ""))
                return f"{b:+.4f}{star_s}"
            print(f"  {s['col']} {s['name']:24s} inter={fmt(r['inter_b'], r['inter_p'])}  "
                  f"(SE {r['inter_se']:.4f}, p={r['inter_p']:.4f})")


if __name__ == "__main__":
    main()
