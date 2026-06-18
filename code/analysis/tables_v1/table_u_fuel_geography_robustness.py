"""Table U: fuel-cut geography robustness and pre/post balance.

The fuel triple is heat x urban_vehicle_hh_ifls4 x post_subsidy within IFLS5, where the
treatment (urban AND vehicle-owning at the IFLS4 baseline) is fixed and post_subsidy
varies by interview date (cut = 2014-11-18). Concern: IFLS5 fielding was not random
across regions over time, so post correlates with geography and the baseline treatment
correlates with geography (urban) -- the interaction could be a region-by-post artifact.

  Panel A -- the triple under a geography fixed-effect ladder, up to geography x post FE
    (which absorbs any region-specific pre/post shift; community-level kecamatan x post is
    the level at which fielding timing varies). The treatment is kept intact -- NOT split
    into urban vs vehicle.
  Panel B -- pre vs post-subsidy balance within IFLS5 on the treatment and on demographics,
    income, and heat (does post track who was interviewed?).
  Panel C -- placebo-in-time. Fake cutoff dates inside the genuinely pre-hike window
    (Sep 6 - Nov 17, 2014, before the real 2014-11-18 cut) should yield no triple. A
    null here argues against the seasonal-confound story. The window is short (~10 weeks,
    ~6k obs) so this bounds large confounds, not effects of the headline size.

Heat = tmean_7d_dev; kabupaten-clustered SE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyfixest as pf

from _lettered_common import (
    CLUSTER,
    CONTROLS,
    KECAMATAN_FE,
    load_analysis,
    require_columns,
    restrict_panel,
    stars,
    write_outputs,
)

TABLE = "table_u_fuel_geography_robustness"
HEAT = "tmean_7d_dev"
UV = "urban_vehicle_hh_ifls4"
POST = "post_subsidy"
TRIPLE_TERMS = [f"{HEAT}:{UV}:{POST}", f"{HEAT}:{POST}:{UV}", f"{POST}:{UV}:{HEAT}"]


def prepare() -> pd.DataFrame:
    df = restrict_panel(load_analysis())
    require_columns(df, [HEAT, UV, POST, KECAMATAN_FE, CLUSTER, "province_code", "urban", "pce_log", "interview_datetime"])
    df = df.dropna(subset=["cesd_z", HEAT, POST, KECAMATAN_FE, CLUSTER, "month", "year", "interview_datetime"]).copy()
    df[UV] = df[UV].fillna(0).astype(int)
    s5 = df[df["wave"] == "IFLS5"].copy()
    s5["idate"] = pd.to_datetime(s5["interview_datetime"])
    s5["province_fe"] = pd.to_numeric(s5["province_code"], errors="coerce").astype("Int64").astype("string")
    s5["kab_post"] = s5[CLUSTER].astype(str) + "_" + s5[POST].astype(int).astype(str)
    s5["kec_post"] = s5[KECAMATAN_FE].astype(str) + "_" + s5[POST].astype(int).astype(str)
    return s5


def fit_triple(s5: pd.DataFrame, fe: str) -> dict[str, float]:
    model = pf.feols(f"cesd_z ~ {HEAT}*{UV}*{POST} + {CONTROLS} | {fe}", s5, vcov={"CRV1": CLUSTER})
    term = next((t for t in TRIPLE_TERMS if t in model.coef().index), None)
    return {
        "n": int(model._N),
        "b": float(model.coef().get(term, np.nan)),
        "se": float(model.se().get(term, np.nan)),
        "p": float(model.pvalue().get(term, np.nan)),
    }


REAL_CUTOFF = "2014-11-18"  # actual fuel-subsidy hike
FAKE_CUTOFFS = ["2014-10-01", "2014-10-10", "2014-10-20"]  # inside the pre-hike window
PLACEBO_TERMS = [f"{HEAT}:{UV}:fpost", f"{HEAT}:fpost:{UV}", f"fpost:{UV}:{HEAT}"]


def fit_cutoff(s5: pd.DataFrame, cutoff: str, pre_window: bool) -> dict[str, float]:
    """Triple at a date cutoff. pre_window restricts to interviews before the real hike
    (so a fake cutoff carries no real policy change); kecamatan FE only given the short span."""
    sub = s5[s5["idate"] < pd.Timestamp(REAL_CUTOFF)] if pre_window else s5
    sub = sub.copy()
    sub["fpost"] = (sub["idate"] >= pd.Timestamp(cutoff)).astype(int)
    fe = KECAMATAN_FE if pre_window else f"month + year + {KECAMATAN_FE}"
    model = pf.feols(f"cesd_z ~ {HEAT}*{UV}*fpost + {CONTROLS} | {fe}", sub, vcov={"CRV1": CLUSTER})
    term = next((t for t in PLACEBO_TERMS if t in model.coef().index), None)
    return {
        "n": int(model._N),
        "b": float(model.coef().get(term, np.nan)),
        "se": float(model.se().get(term, np.nan)),
        "p": float(model.pvalue().get(term, np.nan)),
    }


def balance_row(s5: pd.DataFrame, var: str) -> dict[str, float]:
    sub = s5.dropna(subset=[var]).copy()
    sub[var] = pd.to_numeric(sub[var], errors="coerce")
    sub = sub.dropna(subset=[var])
    pre = sub.loc[sub[POST] == 0, var]
    post = sub.loc[sub[POST] == 1, var]
    pooled_sd = np.sqrt((pre.var() + post.var()) / 2)
    nd = (post.mean() - pre.mean()) / pooled_sd if pooled_sd > 0 else np.nan
    p = float(pf.feols(f"{var} ~ {POST}", sub, vcov={"CRV1": CLUSTER}).pvalue().get(POST, np.nan))
    return {"pre": float(pre.mean()), "post": float(post.mean()), "nd": float(nd), "p": p}


def coef_cell(stats: dict[str, float]) -> str:
    if pd.isna(stats["b"]):
        return r"\textemdash"
    return f"${stats['b']:+.3f}{stars(stats['p'])}$ ({stats['se']:.3f})"


FE_LADDER = [
    ("Month + year (no geography)", "month + year"),
    ("\\quad + province FE", f"month + year + province_fe"),
    ("\\quad + kabupaten FE", f"month + year + {CLUSTER}"),
    ("\\quad + kecamatan FE \\; [headline]", f"month + year + {KECAMATAN_FE}"),
    ("\\quad + kabupaten $\\times$ post FE", f"month + year + {CLUSTER} + kab_post"),
    ("\\quad + kecamatan $\\times$ post FE", "month + year + kec_post"),
]

BALANCE_VARS = [
    (UV, "Urban-vehicle HH (treatment)"),
    ("urban", "Urban"),
    ("pce_log", "Log per-capita expenditure"),
    ("age", "Age"),
    ("female", "Female"),
    ("edu_yrs", "Education (years)"),
    ("married", "Married"),
    ("widowed", "Widowed"),
    (HEAT, "Heat (7-day mean dev.)"),
    ("cesd_z", "CES-D z-score"),
]


def main() -> None:
    s5 = prepare()
    n_pre = int((s5[POST] == 0).sum())
    n_post = int((s5[POST] == 1).sum())

    rows: list[dict] = []
    body = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\multicolumn{5}{l}{\textit{Panel A. Heat $\times$ urban-vehicle $\times$ post under a geography-FE ladder}} \\",
        r"Fixed effects & Triple coefficient & $N$ & & \\",
        r"\midrule",
    ]
    for label, fe in FE_LADDER:
        stats = fit_triple(s5, fe)
        rows.append({"panel": "A", "spec": label, **stats})
        body.append(rf"{label} & {coef_cell(stats)} & {stats['n']:,} & & \\")

    body += [
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Panel B. Pre vs post-subsidy balance within IFLS5}} \\",
        r"Variable & Pre & Post & Norm. diff & Clust. $p$ \\",
        r"\midrule",
    ]
    for var, label in BALANCE_VARS:
        if var not in s5.columns:
            continue
        b = balance_row(s5, var)
        rows.append({"panel": "B", "var": var, "label": label, **b})
        body.append(rf"{label} & {b['pre']:.3f} & {b['post']:.3f} & ${b['nd']:+.3f}$ & {b['p']:.3f} \\")

    body += [
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Panel C. Placebo-in-time: fake cutoff dates with no real hike}} \\",
        r"Cutoff date & Triple coefficient & $N$ & & \\",
        r"\midrule",
    ]
    real = fit_cutoff(s5, REAL_CUTOFF, pre_window=False)
    rows.append({"panel": "C", "cutoff": f"{REAL_CUTOFF} (real)", **real})
    body.append(rf"{REAL_CUTOFF} (real, full IFLS5) & {coef_cell(real)} & {real['n']:,} & & \\")
    for fc in FAKE_CUTOFFS:
        stats = fit_cutoff(s5, fc, pre_window=True)
        rows.append({"panel": "C", "cutoff": f"{fc} (fake)", **stats})
        body.append(rf"{fc} (fake, pre-hike window) & {coef_cell(stats)} & {stats['n']:,} & & \\")

    body += [
        r"\midrule",
        rf"Observations (pre / post) & \multicolumn{{4}}{{c}}{{{n_pre:,} / {n_post:,}}} \\",
        r"\multicolumn{5}{l}{\footnotesize Treatment kept intact (urban $\times$ vehicle, IFLS4 baseline); kabupaten-clustered SE.} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
