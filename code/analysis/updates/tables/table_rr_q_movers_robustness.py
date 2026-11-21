"""Re-run Table B (temperature x shock effects) on location-robustness samples.

Run from the repo root:

    uv run python code/analysis/updates/tables/table_rr_q_movers_robustness.py

Writes output/tables/table_rr_q_movers_robustness.tex. Columns are the seven
Table B specifications (7-day mean temperature only). Each panel re-runs them
on a different version of the data:

    A. Baseline: Table B as is. Each person-wave row carries the household's
       kecamatan in that wave, so cross-wave movers sit in different kecamatan
       fixed-effect cells in IFLS4 and IFLS5.
    B. Drop movers: drop every person whose kecamatan differs between IFLS4 and
       IFLS5. People observed in one wave only are kept. Some "moves" may be
       artefacts of converting 2007 kecamatan codes to 2014 codes.
    C. Drop coarse temperature matches: drop rows whose kecamatan had no GADM
       kecamatan polygon, so temperature was taken from the kabupaten polygon
       (or the province polygon, a handful of cases); see match_level in
       data/02_build_geography.py.
    D. IFLS4 kecamatan: kecamatan fixed effects and kabupaten clusters use
       each person's IFLS4 location in both waves. Temperature is still
       measured where the person lived at each interview. Only run for the
       palm IFLS4-vs-IFLS5 diff-in-diff columns (1-3); the fuel-cut and job
       loss columns identify off within-wave variation, so a cross-wave move
       does not change their kecamatan comparison and the cells are left blank.

The wild cluster bootstrap for Table B lives in
analysis/updates/one-off/wild_cluster_bootstrap_table_b.R.
"""

from dataclasses import replace

import pandas as pd
from analysis.tables_v2.table_b_temperature_and_shock_effects import (
    KABUPATEN_CLUSTER,
    TABLE_SPECS,
    coefficient_stats,
    standardize_model_terms,
)
from library.caching import run_regression_with_caching
from library.config import TABLE_OUTPUT
from library.specs import analysis_df, update_formula_search_replace
from library.table_builder import make_row, stars

TABLE_FILE = TABLE_OUTPUT / "table_rr_q_movers_robustness.tex"
KECAMATAN_FE = "gadm_fullcode"
IFLS4_KECAMATAN = "kecamatan_ifls4"
IFLS4_KABUPATEN = "kabupaten_ifls4"

TABLE_TEMPLATE = r"""
\begin{tabular}{@{}lccccccc}
\toprule
 & \multicolumn{7}{c}{CES-D z-score} \\
\cmidrule(lr){2-8}
 & Palm Shock & \shortstack{Palm Shock\\Panel} & \shortstack{Palm Shock\\Price Drop} & Urban Vehicle Owners & \shortstack{Urban Vehicle Owners\\No Cash Transfer} & \shortstack{Urban Vehicle Owners\\Cash Transfer} & Job Loss \\
\cmidrule(lr){2-2} \cmidrule(lr){3-3} \cmidrule(lr){4-4} \cmidrule(lr){5-5} \cmidrule(lr){6-6} \cmidrule(lr){7-7} \cmidrule(lr){8-8}
 & (1) & (2) & (3) & (4) & (5) & (6) & (7) \\
\midrule\addlinespace[2.5pt]
{panels}
Kecamatan & x & x & x & x & x & x & x \\
Month-Year & x & x & x & x & x & x & x \\
Individual & - & x & - & - & - & - & - \\
\bottomrule
\end{tabular}
"""


# ---------------------------------------------------------------------------
# Data variations
# ---------------------------------------------------------------------------


def cross_wave_movers(df: pd.DataFrame) -> pd.Index:
    """pidlinks whose kecamatan differs between IFLS4 and IFLS5."""
    kecamatans_per_person = df.groupby("pidlink")[KECAMATAN_FE].nunique()
    return kecamatans_per_person.index[kecamatans_per_person > 1]


def ifls4_locations(df: pd.DataFrame) -> pd.DataFrame:
    """One row per person with their IFLS4 kecamatan and kabupaten."""
    return df.loc[
        df["wave"] == "IFLS4", ["pidlink", KECAMATAN_FE, KABUPATEN_CLUSTER]
    ].rename(columns={KECAMATAN_FE: IFLS4_KECAMATAN, KABUPATEN_CLUSTER: IFLS4_KABUPATEN})


MOVERS = cross_wave_movers(analysis_df)
IFLS4_LOCATIONS = ifls4_locations(analysis_df)


def drop_movers(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["pidlink"].isin(MOVERS)].copy()


def drop_coarse_temperature_matches(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["match_level"] == "kecamatan"].copy()


def add_ifls4_location(df: pd.DataFrame) -> pd.DataFrame:
    with_location = df.merge(
        IFLS4_LOCATIONS, on="pidlink", how="left", validate="many_to_one"
    )
    with_location = with_location.dropna(subset=[IFLS4_KECAMATAN]).copy()
    with_location[IFLS4_KABUPATEN] = with_location[IFLS4_KABUPATEN].astype(int)
    return with_location


def with_sample(specs: list[dict], transform) -> list[dict]:
    """Apply a row filter/transform to every spec's data, keeping formulas."""
    return [
        {**spec_data, "spec": replace(spec_data["spec"], df=transform(spec_data["spec"].df))}
        for spec_data in specs
    ]


def is_cross_wave_diff_in_diff(spec_data: dict) -> bool:
    return spec_data["post"] == "ifls5"


def with_ifls4_location(specs: list[dict]) -> list[dict]:
    """Swap the kecamatan fixed effect for the person's IFLS4 kecamatan.

    Only for cross-wave diff-in-diff specs; other specs get spec=None (blank).
    """
    updated = []
    for spec_data in specs:
        if not is_cross_wave_diff_in_diff(spec_data):
            updated.append({**spec_data, "spec": None})
            continue
        spec = update_formula_search_replace(
            spec_data["spec"], KECAMATAN_FE, IFLS4_KECAMATAN
        )
        updated.append({**spec_data, "spec": replace(spec, df=add_ifls4_location(spec.df))})
    return updated


# Each panel: (label, specs, cluster variable).
PANELS = [
    ("Panel A: Baseline", TABLE_SPECS, KABUPATEN_CLUSTER),
    ("Panel B: Drop cross-wave movers", with_sample(TABLE_SPECS, drop_movers), KABUPATEN_CLUSTER),
    (
        "Panel C: Drop kabupaten/province temperature matches",
        with_sample(TABLE_SPECS, drop_coarse_temperature_matches),
        KABUPATEN_CLUSTER,
    ),
    ("Panel D: IFLS4 kecamatan fixed effects", with_ifls4_location(TABLE_SPECS), IFLS4_KABUPATEN),
]


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------


def differential_term(model) -> str:
    return "group:post:heat" if "group:post:heat" in model.coef_table.index else "group:heat"



def run_panel(specs: list[dict], cluster: str) -> list[dict]:
    """Fit each spec and pull out its differential-heat term.

    Identification (as in Table B): kecamatan and month-by-year fixed effects
    (plus individual FE in the panel column), so heat effects come from 7-day
    weather deviations around local and calendar norms. The differential term
    assumes those deviations are as good as random with respect to shock
    status within kecamatan and month-year.
    """
    results = []
    for spec_data in specs:
        spec = spec_data["spec"]
        if spec is None:
            results.append(None)
            continue
        fit = run_regression_with_caching(spec, vcov_type=cluster)
        replacements = {spec_data["group"]: "group", spec_data["heat"]: "heat"}
        if spec_data["post"] is not None:
            replacements[spec_data["post"]] = "post"
        model = standardize_model_terms(fit, replacements)
        results.append({"model": model, "term": differential_term(model)})
    return results


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def coefficient_row_pair(label: str, results: list[dict | None], term_of) -> tuple[str, str]:
    """Coefficient and (SE) rows; columns without a model are left blank."""
    coefficient_cells, standard_error_cells = [], []
    for r in results:
        if r is None:
            coefficient_cells.append("")
            standard_error_cells.append("")
            continue
        coefficient, standard_error, p_value = coefficient_stats(r["model"], term_of(r))
        coefficient_cells.append(f"{coefficient:.3f}{stars(p_value)}")
        standard_error_cells.append(f"({standard_error:.3f})")
    return make_row(label, coefficient_cells), make_row("", standard_error_cells)


def panel_rows(label: str, results: list[dict | None]) -> str:
    differential_coefs, differential_ses = coefficient_row_pair(
        "Differential Impact of Heat", results, lambda r: r["term"]
    )
    heat_coefs, heat_ses = coefficient_row_pair("Heat", results, lambda r: "heat")
    observations_row = make_row(
        "Observations",
        [f"{int(r['model'].stats['N']):,}" if r else "" for r in results],
    )
    return "\n".join(
        [
            rf"\multicolumn{{8}}{{l}}{{\textit{{{label}}}}} \\",
            r"\midrule",
            differential_coefs,
            differential_ses,
            heat_coefs,
            heat_ses,
            observations_row,
            r"\midrule\addlinespace[2.5pt]",
        ]
    )


def summary_frame(label: str, results: list[dict | None]) -> pd.DataFrame:
    rows = []
    for spec_data, r in zip(TABLE_SPECS, results):
        if r is None:
            continue
        b, se, p = coefficient_stats(r["model"], r["term"])
        rows.append(
            {
                "panel": label.split(":")[0],
                "column": spec_data["label"]
                .replace(r"\shortstack{", "")
                .replace(r"\\", " ")
                .replace("}", ""),
                "b": b,
                "se": se,
                "p": p,
                "N": int(r["model"].stats["N"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    print(f"Cross-wave movers (kecamatan changed): {len(MOVERS):,} people")
    panel_texts, summaries = [], []
    for label, specs, cluster in PANELS:
        results = run_panel(specs, cluster)
        panel_texts.append(panel_rows(label, results))
        summaries.append(summary_frame(label, results))

    TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TABLE_FILE.write_text(
        TABLE_TEMPLATE.replace("{panels}", "\n".join(panel_texts)), encoding="utf-8"
    )
    with pd.option_context("display.width", 200, "display.precision", 4):
        print("Differential impact of heat (tmean_7d):")
        print(pd.concat(summaries, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
