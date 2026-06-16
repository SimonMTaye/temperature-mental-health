"""Table Q: continuous palm-intensity dose-response.

Replaces the binary palm-farmer x IFLS5 shock with continuous provincial palm
dependence (BPS planted area, wave-matched 2007/2014). If the heat amplification
scales with how palm-dependent the province is -- and is absent where palm acreage
is low -- it is much harder to attribute to "something else changed in 2014."

No kabupaten-level palm-area file exists, so intensity is province-level. Provincial
palm area is read from data/raw/palm_area_prov_BPS.csv (BPS / Ditjenbun, Statistik
Perkebunan: Kelapa Sawit), repo-local external data. Heat measure: tmean_7d_dev.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _lettered_common import (
    CONTROL_COLUMNS,
    CONTROLS,
    FE_POOLED,
    KECAMATAN_FE,
    PROJECT,
    cell,
    fit_model,
    load_analysis,
    restrict_panel,
    term_stats,
    write_outputs,
)

TABLE = "table_q_palm_intensity_dose"
HEAT = "tmean_7d_dev"
PALM_AREA_CSV = PROJECT / "data" / "raw" / "palm_area_prov_BPS.csv"


def add_palm_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Merge provincial palm acreage (wave-matched) and build continuous + tercile intensity."""
    pa = pd.read_csv(PALM_AREA_CSV).rename(columns={"prov_code": "province_code"})
    df = df.merge(pa[["province_code", "TOT_area_2007", "TOT_area_2014"]], on="province_code", how="left")
    df["palm_area_ha"] = np.where(df["wave"] == "IFLS4", df["TOT_area_2007"], df["TOT_area_2014"]).astype(float)
    df["palm_area_ha"] = df["palm_area_ha"].fillna(0.0)

    log_area = np.log1p(df["palm_area_ha"])
    df["palm_int_z"] = (log_area - log_area.mean()) / log_area.std()

    # Terciles of provincial palm acreage among palm provinces; 0 = non-palm base.
    df["palm_bin"] = 0
    pos = df["palm_area_ha"] > 0
    cut = df.loc[pos, "palm_area_ha"]
    t1, t2 = cut.quantile(1 / 3), cut.quantile(2 / 3)
    df.loc[pos & (df["palm_area_ha"] <= t1), "palm_bin"] = 1
    df.loc[pos & (df["palm_area_ha"] > t1) & (df["palm_area_ha"] <= t2), "palm_bin"] = 2
    df.loc[pos & (df["palm_area_ha"] > t2), "palm_bin"] = 3
    for b in (1, 2, 3):
        df[f"palm_t{b}"] = (df["palm_bin"] == b).astype(int)
    return df


def fit_triple(df: pd.DataFrame, treatments: list[str]):
    """Fit heat x IFLS5 x treatment(s); returns the fitted model."""
    rhs = " + ".join(f"{HEAT}*ifls5*{t}" for t in treatments)
    formula = f"cesd_z ~ {rhs} + {CONTROLS} | {FE_POOLED}"
    required = ["cesd_z", HEAT, "ifls5", *treatments, "month", "year", "wave", KECAMATAN_FE, *CONTROL_COLUMNS]
    return fit_model(df, formula, required)


def main() -> None:
    df = add_palm_intensity(restrict_panel(load_analysis()))

    specs = []
    m_bin = fit_triple(df, ["palm_farmer_hh_ifls4"])
    specs.append(("Binary palm farmer $\\times$ IFLS5 (reference)", term_stats(m_bin, f"{HEAT}:ifls5:palm_farmer_hh_ifls4")))
    m_cont = fit_triple(df, ["palm_int_z"])
    specs.append(("Palm intensity (z) $\\times$ IFLS5 \\; [continuous dose]", term_stats(m_cont, f"{HEAT}:ifls5:palm_int_z")))
    m_bins = fit_triple(df, ["palm_t1", "palm_t2", "palm_t3"])
    for b, lab in [(1, "T1 (low acreage)"), (2, "T2 (mid)"), (3, "T3 (high acreage)")]:
        specs.append((f"\\quad Palm acreage {lab} $\\times$ IFLS5", term_stats(m_bins, f"{HEAT}:ifls5:palm_t{b}")))

    rows = [{"spec": label, **stats} for label, stats in specs]
    body = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Treatment intensity & Heat $\times$ IFLS5 $\times$ intensity & $N$ \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Dependent variable: CES-D z-score; heat = 7-day mean deviation}} \\",
        r"\addlinespace[2pt]",
    ]
    for label, stats in specs:
        coef_cell, se_cell = cell(stats["b"], stats["se"], stats["p"])
        body.append(rf"{label} & {coef_cell} & {int(stats['n']):,} \\")
        body.append(rf" & {se_cell} & \\")
        body.append(r"\addlinespace[2pt]")
    body.extend(
        [
            r"\midrule",
            r"Demographic controls & Yes & \\",
            r"Kecamatan + Month + Year + Wave FE & Yes & \\",
            r"Cluster: kabupaten & Yes & \\",
            r"Sample & Panel & \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
