"""Table R: identify the palm shock off the interview-date price path, not the wave dummy.

IFLS5 fielding ran 2014-09 -> 2015-09 while palm prices fell (715 -> 511 USD/MT), and
IFLS4 (2007-08) spanned an even larger price swing. Because interview timing varies,
the palm price each household faced varies. This table replaces the binary IFLS5 dummy
in the palm triple with the actual palm price at interview, in several forms, to see
whether the effect survives identification off the price itself.

Honest finding (state plainly in the paper): the powered, pooled price-at-interview
specs reproduce the effect, but the variation that is clean of the wave -- the
within-IFLS5 price and the household IFLS4->IFLS5 price gap -- is null. Palm price is
largely collinear with the wave, so price-timing alone does not separately identify it.

palm_price_usd_mt / palm_price_z are already in the canonical input. Heat: tmean_7d_dev.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyfixest as pf

from _lettered_common import (
    CLUSTER,
    CONTROLS,
    KECAMATAN_FE,
    cell,
    load_analysis,
    require_columns,
    restrict_panel,
    write_outputs,
)

TABLE = "table_r_palm_price_path"
HEAT = "tmean_7d_dev"
PALM = "palm_farmer_hh_ifls4"
FE_POOLED = f"month + year + wave + {KECAMATAN_FE}"
FE_IFLS5 = f"month + year + {KECAMATAN_FE}"
FE_NOWAVE = f"month + year + {KECAMATAN_FE}"
FE_PIDLINK = f"pidlink + month + {KECAMATAN_FE}"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-shock encodings derived from the palm price at interview."""
    require_columns(df, ["palm_price_usd_mt", "palm_price_z", "interview_datetime"])
    df = df.dropna(subset=["palm_price_usd_mt", "palm_price_z"]).copy()
    df["price_decline"] = -df["palm_price_z"]  # higher = lower price = bigger shock
    neg = -df["palm_price_usd_mt"]
    df["negprice_z"] = (neg - neg.mean()) / neg.std()
    # household IFLS4 -> IFLS5 price gap (varies with interview timing)
    w4 = df[df["wave"] == "IFLS4"].groupby("pidlink")["palm_price_usd_mt"].first()
    w5 = df[df["wave"] == "IFLS5"].groupby("pidlink")["palm_price_usd_mt"].first()
    gap = (w4 - w5).rename("price_gap")
    df = df.merge(gap, on="pidlink", how="left")
    df["price_gap_z"] = (df["price_gap"] - df["price_gap"].mean()) / df["price_gap"].std()
    return df


def fit(data: pd.DataFrame, shock: str, fe: str) -> dict[str, float]:
    formula = f"cesd_z ~ {HEAT}*{PALM}*{shock} + {CONTROLS} | {fe}"
    model = pf.feols(formula, data=data, vcov={"CRV1": CLUSTER})
    term = f"{HEAT}:{PALM}:{shock}"
    return {
        "n": int(model._N),
        "b": float(model.coef().get(term, np.nan)),
        "se": float(model.se().get(term, np.nan)),
        "p": float(model.pvalue().get(term, np.nan)),
    }


def main() -> None:
    df = prepare(restrict_panel(load_analysis()))
    df[PALM] = df[PALM].fillna(0).astype(int)
    ifls5 = df[df["wave"] == "IFLS5"].copy()

    # (label, identifying variation, estimation sample, shock var, FE)
    specs = [
        ("Binary IFLS5 wave dummy (reference)", "between-wave", df, "ifls5", FE_POOLED),
        ("Palm price decline, within IFLS5", "within-wave timing", ifls5, "price_decline", FE_IFLS5),
        ("Price at interview, pooled (no wave FE)", "cross-wave price level", df, "negprice_z", FE_NOWAVE),
        ("Price at interview, pooled + household FE", "within-household price", df, "negprice_z", FE_PIDLINK),
        ("IFLS4$\\to$IFLS5 household price gap", "interview-timing gap", ifls5, "price_gap_z", FE_IFLS5),
    ]

    rows = []
    body = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Price encoding & Identifying variation & Heat $\times$ palm $\times$ shock & $N$ \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Dependent variable: CES-D z-score; heat = 7-day mean deviation}} \\",
        r"\addlinespace[2pt]",
    ]
    for label, idvar, data, shock, fe in specs:
        stats = fit(data, shock, fe)
        rows.append({"spec": label, "idvar": idvar, "shock": shock, **stats})
        coef_cell, se_cell = cell(stats["b"], stats["se"], stats["p"])
        body.append(rf"{label} & {idvar} & {coef_cell} & {int(stats['n']):,} \\")
        body.append(rf" & & {se_cell} & \\")
        body.append(r"\addlinespace[2pt]")
    body.extend(
        [
            r"\midrule",
            r"Demographic controls & \multicolumn{3}{c}{Yes} \\",
            r"Cluster: kabupaten & \multicolumn{3}{c}{Yes} \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
