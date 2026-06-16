"""Table P: heat x job-loss with slope controls.

Job loss is endogenous: people who recently lost work are not a random draw, so
the heat x job_loss_1_yr interaction may pick up heat-sensitivity that travels
with occupation, outdoor exposure, income (cooling capacity), or pre-existing
health rather than the labour-market event itself. A construction labourer who
just lost a job is, in the naive spec, compared with an office worker who did
not -- their heat slopes differ for reasons unrelated to unemployment.

The fix is to absorb the *slope* of these confounders, not just their level: add
heat-interacted controls and watch whether the heat x job_loss coefficient
survives. Stability => a real amplification effect; collapse => the baseline was
a slope confound.

Slope controls added in sequence:
  S1. Heat x IFLS4 sector dummies (9)  -- occupation-specific heat sensitivity;
      identifies job-loss x heat within occupation type.
  S2. Heat x outdoor work (agri/mining/construction) -- exposure / cooling-access proxy.
  S3. Heat x baseline health (many_symptoms, recent_hospitalised, recent_accident)
      -- pre-existing / chronic health vulnerability. (No direct chronic-condition
      field in the canonical input; these are the available health proxies.)
  S4. Heat x age -- life-cycle heat sensitivity.
  S5. Heat x log PCE -- income / cooling-capacity (AC-access) proxy.
  S6. All of the above jointly.

TODO(AC): the most direct cooling-access control is household air-conditioning
ownership. AC is present in the raw IFLS household assets module but is NOT yet
in the cleaned 30_analysis_table_input.parquet. Once it is cleaned in (e.g. an
``ac_ownership`` / ``has_ac`` column), add a ``Heat x AC access`` slope-control
spec here and fold it into the S6 joint spec -- it is a better cooling proxy
than log PCE (S5), which is the stand-in until then.

IFLS4 sector is carried forward from raw IFLS b3a_tk2 (tk19ab), read from the
folders configured in code/data/config.py. Heat measure: tmean_7d_dev.
"""

from __future__ import annotations

import sys

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
    restrict_ifls4_panel,
    term_stats,
    write_outputs,
)

# Raw IFLS sidecar folders live in the data pipeline's central path config.
sys.path.insert(0, str(PROJECT / "code" / "data"))
from config import IFLS4_FOLDER  # noqa: E402

TABLE = "table_p_jobloss_slope_controls"
HEAT = "tmean_7d_dev"
TERM = f"{HEAT}:job_loss_1_yr"

SECTOR_COLS = [
    "sec_agri", "sec_mining", "sec_manuf", "sec_util", "sec_constr",
    "sec_trade", "sec_transp", "sec_finance", "sec_services",
]
HEALTH_COLS = ["many_symptoms", "recent_hospitalised", "recent_accident_2y"]


def load_ifls4_sector() -> pd.DataFrame:
    """IFLS4-baseline 1-digit employment sector (tk19ab), carried forward by pidlink."""
    tk = pd.read_stata(
        IFLS4_FOLDER / "b3a_tk2.dta", convert_categoricals=False, columns=["pidlink", "tk19ab"]
    )
    sector = pd.to_numeric(tk["tk19ab"].astype(str).str.strip(), errors="coerce")
    return (
        pd.DataFrame({"pidlink": tk["pidlink"], "ifls4_sector": sector.astype("Int64")})
        .drop_duplicates("pidlink")
    )


def build_slope_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Add heat-slope confounders: sector dummies, outdoor, health, centered age/PCE."""
    df = df.merge(load_ifls4_sector(), on="pidlink", how="left")
    df["ifls4_sector"] = pd.to_numeric(df["ifls4_sector"], errors="coerce").fillna(0).astype(int)

    # Sector 0 (missing) is the omitted base; sectors follow the IFLS 1-digit codes.
    for code, col in enumerate(SECTOR_COLS, start=1):
        df[col] = (df["ifls4_sector"] == code).astype(int)
    df["ifls4_outdoor"] = df["ifls4_sector"].isin([1, 2, 5]).astype(int)  # agri, mining, construction

    for col in HEALTH_COLS:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

    df["job_loss_1_yr"] = df["job_loss_1_yr"].fillna(0).astype(int)
    df["pce_log"] = pd.to_numeric(df["pce_log"], errors="coerce")
    df["pce_log"] = df["pce_log"].fillna(df["pce_log"].median())
    df["age_dev"] = df["age"] - df["age"].mean()
    df["pce_log_dev"] = df["pce_log"] - df["pce_log"].mean()
    return df


def heat_interactions(columns: list[str]) -> str:
    """Formula fragment interacting each confounder with heat."""
    return " + ".join(f"{HEAT} * {col}" for col in columns)


def fit_spec(df: pd.DataFrame, slope_columns: list[str]) -> dict[str, object]:
    """Fit cesd_z ~ heat * job_loss + heat-slope controls and return the interaction stats."""
    slope = heat_interactions(slope_columns)
    extra = f" + {slope}" if slope else ""
    formula = f"cesd_z ~ {HEAT} * job_loss_1_yr{extra} + {CONTROLS} | {FE_POOLED}"
    required = [
        "cesd_z", HEAT, "job_loss_1_yr",
        "month", "year", "wave", KECAMATAN_FE,
        *CONTROL_COLUMNS, *slope_columns,
    ]
    model = fit_model(df, formula, required)
    return term_stats(model, TERM)


# (row label, columns interacted with heat as slope controls)
# NOTE(AC): add ("S7. + Heat $\\times$ AC access", ["has_ac"]) and append the AC
# column to the S6 joint list once AC ownership is in the canonical input.
SPECS = [
    ("S0. None (baseline)", []),
    ("S1. + Heat $\\times$ occupation sector (9)", SECTOR_COLS),
    ("S2. + Heat $\\times$ outdoor work", ["ifls4_outdoor"]),
    ("S3. + Heat $\\times$ baseline health", HEALTH_COLS),
    ("S4. + Heat $\\times$ age", ["age_dev"]),
    ("S5. + Heat $\\times$ log PCE (cooling capacity)", ["pce_log_dev"]),
    ("S6. + All slope controls jointly", SECTOR_COLS + HEALTH_COLS + ["age_dev", "pce_log_dev"]),
]


def main() -> None:
    df = build_slope_controls(restrict_ifls4_panel(load_analysis()))
    rows = [{"spec": label, "slope": "+".join(cols) or "none", **fit_spec(df, cols)} for label, cols in SPECS]

    body = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Slope controls added & Heat $\times$ job loss & $N$ \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Dependent variable: CES-D z-score; heat = 7-day mean deviation}} \\",
        r"\addlinespace[2pt]",
    ]
    for row in rows:
        coef_cell, se_cell = cell(row["b"], row["se"], row["p"])
        body.append(rf"{row['spec']} & {coef_cell} & {int(row['n']):,} \\")
        body.append(rf" & {se_cell} & \\")
        body.append(r"\addlinespace[2pt]")
    body.extend(
        [
            r"\midrule",
            r"Demographic controls & Yes & \\",
            r"Kecamatan + Month + Year + Wave FE & Yes & \\",
            r"Cluster: kabupaten & Yes & \\",
            r"Sample & IFLS4-baseline panel & \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_outputs(TABLE, body, rows)


if __name__ == "__main__":
    main()
