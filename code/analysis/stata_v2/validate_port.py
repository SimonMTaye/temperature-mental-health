"""Compare the Stata port (esttab CSVs) against fresh pyfixest fits on the same single dataset.

Run from the repository root after export_port_data.py and port_tables.do.
No regression cache is read; every model is re-fit here.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import pyfixest as pf

from analysis.tables_v2 import table_a_temperature_effects as a
from analysis.tables_v2 import table_b_temperature_and_shock_effects as b
from analysis.tables_v2 import table_d_palm_farmer_alt_definitions as d
from library.port_helper import EXPORT_DIR, EXPORT_STEM
from library.specs import build_palm_df, temperature_spec, update_formula_search_replace, wave5_df

STATA_TABLES = Path("output/tables/stata_port")
DELIVERY = Path(".cache/stata_port/validation")
TOLERANCES = {"b": 1e-7, "se": 1e-7, "N": 0, "N_clust": 0}
CLUSTER = {"CRV1": "kabupaten_full_code"}


def stata_name(column: str) -> str:
    """Exported (Stata-safe) name of a Python column, from the export mapping sidecar."""
    mapping = json.loads((EXPORT_DIR / f"{EXPORT_STEM}_mapping.json").read_text())
    return mapping["column_names"][column]


def displayed_terms(heat, group=None, post=None) -> dict[str, str]:
    """Map pyfixest term names to Stata coefficient names for the displayed terms."""
    terms = {heat: heat}
    if group:
        parts = [x for x in [group, post, heat] if x]
        terms[":".join(parts)] = "#".join(f"c.{stata_name(x)}" for x in parts)
    return terms


def collect_models() -> list[dict]:
    """One entry per ported regression: esttab table, eststo name, formula, sample restriction, terms."""
    models = []
    for table, outcome in [("table_a", "cesd_z"), ("table_a2", "sleep_dur_h")]:
        for rid, (_, heat) in enumerate(a.OVERALL_HEAT_COLUMNS, 1):
            spec = update_formula_search_replace(temperature_spec, "tmean_7d", heat)
            if outcome != "cesd_z":
                spec = update_formula_search_replace(spec, "cesd_z", outcome)
            models.append(dict(table=table, name=f"{table}_{rid}", formula=spec.formula, wave5_only=False, terms=displayed_terms(heat)))
    for rid, item in enumerate(b.TABLE_SPECS + b.wetbulb_specs(), 1):
        table = "table_b_temperature" if rid <= 7 else "table_b_wetbulb"
        models.append(dict(table=table, name=f"table_b_{rid}", formula=item["spec"].formula, wave5_only=item["spec"].df is wave5_df, terms=displayed_terms(item["heat"], item["group"], item["post"])))
    for rid, (spec, (_, group)) in enumerate(zip(d.make_specs(None), d.GROUPS), 1):
        models.append(dict(table="table_d", name=f"table_d_{rid}", formula=spec.formula, wave5_only=False, terms=displayed_terms("tmean_7d", group, "ifls5")))
    return models


def fit_python(models: list[dict], tight: bool) -> pd.DataFrame:
    """Fresh pyfixest fits on the exported dataset; long rows of (table, name, term, metric, python_value)."""
    port_df = build_palm_df()
    options = {"fixef_tol": 1e-10, "fixef_maxiter": 100000} if tight else {}
    rows = []
    for model in models:
        data = port_df[port_df["ifls5"]] if model["wave5_only"] else port_df
        fit = pf.feols(model["formula"], data=data.copy(), vcov=CLUSTER, **options)
        tidy = fit.tidy()
        rows.append(dict(table=model["table"], name=model["name"], term="N", metric="N", python_value=float(fit._N)))
        rows.append(dict(table=model["table"], name=model["name"], term="N_clust", metric="N_clust", python_value=float(fit._G[0])))
        for python_term, stata_term in model["terms"].items():
            rows.append(dict(table=model["table"], name=model["name"], term=stata_term, metric="b", python_value=float(tidy.loc[python_term, "Estimate"])))
            rows.append(dict(table=model["table"], name=model["name"], term=stata_term, metric="se", python_value=float(tidy.loc[python_term, "Std. Error"])))
        print(f'{model["name"]}: N={fit._N}, clusters={fit._G[0]}', flush=True)
    return pd.DataFrame(rows)


def read_esttab_csv(table: str) -> pd.DataFrame:
    """Parse an esttab csv written with cells("b se") into long rows of (table, name, term, metric, stata_value)."""
    raw = pd.read_csv(STATA_TABLES / f"{table}.csv", header=None, dtype=str, keep_default_na=False)
    rows = []
    name = None
    for column in raw.columns[1:]:
        name = raw.loc[0, column] or name
        metric = raw.loc[1, column]
        for term, value in zip(raw.loc[2:, 0], raw.loc[2:, column]):
            if value != "":
                rows.append(dict(table=table, name=name, term=term, metric=term if term in ("N", "N_clust") else metric, stata_value=float(value)))
    return pd.DataFrame(rows)


def compare(python: pd.DataFrame) -> None:
    stata = pd.concat(read_esttab_csv(table) for table in python["table"].unique())
    keys = ["table", "name", "term", "metric"]
    comparison = python.merge(stata, on=keys, how="left", validate="one_to_one")
    comparison["difference"] = comparison["stata_value"] - comparison["python_value"]
    comparison["absolute_tolerance"] = comparison["metric"].map(TOLERANCES)
    comparison["passed"] = comparison["difference"].abs() <= comparison["absolute_tolerance"]
    DELIVERY.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(DELIVERY / "python_stata_comparison.csv", index=False)
    max_abs = comparison.groupby("metric")["difference"].apply(lambda s: float(s.abs().max())).to_dict()
    summary = dict(
        models=int(comparison["name"].nunique()),
        rows=len(comparison),
        passed=int(comparison["passed"].sum()),
        missing_in_stata=int(comparison["stata_value"].isna().sum()),
        samples_matched=int(comparison.query("metric == 'N'")["passed"].sum()),
        max_abs_difference=max_abs,
        tolerances=TOLERANCES,
    )
    (DELIVERY / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    failures = comparison[~comparison["passed"]]
    if len(failures):
        print(failures.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tight", action="store_true", help="Diagnostic: fixef_tol=1e-10, fixef_maxiter=100000; same specification")
    args = parser.parse_args()
    if args.tight:
        DELIVERY = DELIVERY / "tight_projection"
    compare(fit_python(collect_models(), tight=args.tight))
