"""Table M: palm smallholder vs other-palm mechanism.

Decomposes the headline heat x palm_farmer_hh_ifls4 x IFLS5 amplification by
the IFLS4-baseline household agricultural employment status of the HH head.

  palm_self_pure_ifls4  = palm_farmer_hh_ifls4 AND ag_self_2007 AND NOT ag_wage_2007
                          (self-employed smallholder — bears CPO price risk)
  palm_other_ifls4      = palm_farmer_hh_ifls4 AND NOT palm_self_pure_ifls4
                          (wage workers, family workers, mixed, other)

ag_self_2007 / ag_wage_2007 are derived from raw IFLS b3a_tk2 (tk19ab sector + tk24a
working status) at the household level, carried forward from IFLS4. These columns
are NOT in the canonical 30_analysis_table_input.parquet — sidecar read from the
raw IFLS folders configured in code/data/config.py.

Specifications (panel sample, kecamatan FE, kabupaten-clustered SE):
  Heat x palm_self_pure_ifls4 x IFLS5  (separate triple)
  Heat x palm_other_ifls4 x IFLS5      (separate triple)
  Heat x palm_self_pure_ifls4 x IFLS5  (jointly with palm_other)
  Heat x palm_other_ifls4 x IFLS5      (jointly with palm_self_pure)
"""

from __future__ import annotations

import sys

import pandas as pd

from _lettered_common import (
    CONTROLS,
    FE_POOLED,
    PROJECT,
    cell,
    fit_model,
    load_analysis,
    restrict_panel,
    term_stats,
    write_outputs,
)

# Raw IFLS sidecar folders live in the data pipeline's central path config.
sys.path.insert(0, str(PROJECT / "code" / "data"))
from config import IFLS4_FOLDER, IFLS5_FOLDER  # noqa: E402

TABLE = "table_m_palm_smallholder_mechanism"

TK2_FILES = {
    "IFLS4": IFLS4_FOLDER / "b3a_tk2.dta",
    "IFLS5": IFLS5_FOLDER / "b3a_tk2.dta",
}
HH_COL = {"IFLS4": "hhid07", "IFLS5": "hhid14"}


def load_hh_emp_status() -> pd.DataFrame:
    """HH-level agricultural employment indicators per wave."""
    frames = []
    for wave, path in TK2_FILES.items():
        tk = pd.read_stata(path, convert_categoricals=False)
        sec = pd.to_numeric(tk["tk19ab"].astype(str).str.strip(), errors="coerce")
        emp = pd.to_numeric(tk["tk24a"].astype(str).str.strip(), errors="coerce")
        tk["ag_wage"] = ((sec == 1) & emp.isin([5, 7])).astype(int)
        tk["ag_self"] = ((sec == 1) & emp.isin([1, 2, 3])).astype(int)
        hh_col = HH_COL[wave]
        agg = tk.groupby(hh_col).agg(
            hh_has_ag_wage=("ag_wage", "max"),
            hh_has_ag_self=("ag_self", "max"),
        ).reset_index().rename(columns={hh_col: "hhid"})
        agg["hhid"] = agg["hhid"].astype(str)
        agg["wave"] = wave
        frames.append(agg)
    return pd.concat(frames, ignore_index=True)


def build_smallholder_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Merge HH employment status and build IFLS4-baseline smallholder indicators."""
    emp = load_hh_emp_status()
    df = df.copy()
    df["hhid_s"] = df["hhid"].astype(str)
    emp = emp.rename(columns={"hhid": "hhid_s"})
    df = df.merge(emp, on=["hhid_s", "wave"], how="left")
    for col in ("hh_has_ag_wage", "hh_has_ag_self"):
        df[col] = df[col].fillna(0).astype(int)

    base4 = (
        df.loc[df["wave"] == "IFLS4", ["pidlink", "hh_has_ag_wage", "hh_has_ag_self"]]
        .rename(columns={
            "hh_has_ag_wage": "ag_wage_ifls4",
            "hh_has_ag_self": "ag_self_ifls4",
        })
        .drop_duplicates("pidlink")
    )
    df = df.merge(base4, on="pidlink", how="left")
    for col in ("ag_wage_ifls4", "ag_self_ifls4"):
        df[col] = df[col].fillna(0).astype(int)

    palm = df["palm_farmer_hh_ifls4"].fillna(0).astype(int)
    df["palm_self_pure_ifls4"] = (
        palm * df["ag_self_ifls4"] * (1 - df["ag_wage_ifls4"])
    ).astype(int)
    df["palm_other_ifls4"] = (palm * (1 - df["palm_self_pure_ifls4"])).astype(int)
    return df


def fit_triple(df: pd.DataFrame, group: str) -> dict[str, object]:
    formula = (
        f"cesd_z ~ heat_c_dev * ifls5 * {group} + {CONTROLS} | {FE_POOLED}"
    )
    required = [
        "cesd_z",
        "heat_c_dev",
        "ifls5",
        group,
        "month",
        "year",
        "wave",
        "kecamatan_code",
        "kabupaten_code",
        "age",
        "female",
        "edu_yrs",
        "married",
        "widowed",
    ]
    model = fit_model(df, formula, required)
    term = f"heat_c_dev:ifls5:{group}"
    stats = term_stats(model, term)
    return {"label": group, "term": term, **stats}


def fit_joint(df: pd.DataFrame) -> list[dict[str, object]]:
    formula = (
        "cesd_z ~ heat_c_dev * ifls5 * palm_self_pure_ifls4 "
        "+ heat_c_dev * ifls5 * palm_other_ifls4 "
        f"+ {CONTROLS} | {FE_POOLED}"
    )
    required = [
        "cesd_z",
        "heat_c_dev",
        "ifls5",
        "palm_self_pure_ifls4",
        "palm_other_ifls4",
        "month",
        "year",
        "wave",
        "kecamatan_code",
        "kabupaten_code",
        "age",
        "female",
        "edu_yrs",
        "married",
        "widowed",
    ]
    model = fit_model(df, formula, required)
    out = []
    for group in ("palm_self_pure_ifls4", "palm_other_ifls4"):
        term = f"heat_c_dev:ifls5:{group}"
        out.append({"label": f"joint_{group}", "term": term, **term_stats(model, term)})
    return out


def main() -> None:
    df = restrict_panel(load_analysis())
    df = build_smallholder_indicators(df)

    rows: list[dict[str, object]] = []
    rows.append({"spec": "separate", **fit_triple(df, "palm_self_pure_ifls4")})
    rows.append({"spec": "separate", **fit_triple(df, "palm_other_ifls4")})
    for joint in fit_joint(df):
        rows.append({"spec": "joint", **joint})

    sep_self  = rows[0]
    sep_other = rows[1]
    jnt_self  = rows[2]
    jnt_other = rows[3]

    def c(row):
        return cell(row["b"], row["se"], row["p"])

    body = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r" & (1) Separate & (2) Joint \\",
        r" & triple & triples \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Dependent variable: CES-D z-score}} \\",
        r"\addlinespace[2pt]",
        rf"Heat $\times$ IFLS5 $\times$ palm smallholder (self-employed) & {c(sep_self)[0]} & {c(jnt_self)[0]} \\",
        rf" & {c(sep_self)[1]} & {c(jnt_self)[1]} \\",
        r"\addlinespace[2pt]",
        rf"Heat $\times$ IFLS5 $\times$ palm other (wage / family / mixed) & {c(sep_other)[0]} & {c(jnt_other)[0]} \\",
        rf" & {c(sep_other)[1]} & {c(jnt_other)[1]} \\",
        r"\midrule",
        r"Demographic controls & Yes & Yes \\",
        r"Kecamatan FE & Yes & Yes \\",
        r"Month + Year FE & Yes & Yes \\",
        r"Wave FE & Yes & Yes \\",
        r"\addlinespace[3pt]",
        r"Sample & Panel & Panel \\",
        rf"Observations & {int(sep_self['n']):,} & {int(jnt_self['n']):,} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
