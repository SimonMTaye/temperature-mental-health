"""Fresh Python references and numeric/sample comparison for the explicit Stata port.

Run from the repository root after port_tables.do. No existing regression cache is read.
"""
from pathlib import Path
import json
import argparse
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyfixest as pf

from analysis.tables_v2 import table_a_temperature_effects as a
from analysis.tables_v2 import table_b_temperature_and_shock_effects as b
from analysis.tables_v2 import table_d_palm_farmer_alt_definitions as d
from library.specs import temperature_spec, update_formula_search_replace

OUT = Path('.cache/stata_port')
DELIVERY = OUT / 'validation'


def collect_actual_specs():
    models = []
    for code, outcome in [('table_a', 'cesd_z'), ('table_a2', 'sleep_dur_h')]:
        for rid, (_, heat) in enumerate(a.OVERALL_HEAT_COLUMNS, 1):
            spec = update_formula_search_replace(temperature_spec, 'tmean_7d', heat)
            if outcome != 'cesd_z':
                spec = update_formula_search_replace(spec, 'cesd_z', outcome)
            models.append((code, rid, spec, heat, None, None))
    for rid, item in enumerate(b.TABLE_SPECS + b.wetbulb_specs(), 1):
        models.append(('table_b', rid, item['spec'], item['heat'], item['group'], item['post']))
    palm_specs = []
    # Execute D's existing Python preparation; suppress exports/regressions/rendering.
    with patch.object(d, 'save_data'), patch.object(d, 'make_shock_regression_table_trimmed', side_effect=lambda specs, **kwargs: palm_specs.extend(specs)), patch.object(d, 'render_table_to_latex'):
        d.make_table()
    for rid, spec in enumerate(palm_specs, 1):
        group = ['palm_farmer_hh_ifls4', 'palm_farmer_both_waves', 'palm_farmer_hh_ifls5'][rid - 1]
        models.append(('table_d', rid, spec, 'tmean_7d', group, 'ifls5'))
    return models


def estimate_fresh_references(tight=False):
    OUT.mkdir(parents=True, exist_ok=True)
    DELIVERY.mkdir(parents=True, exist_ok=True)
    rows = []
    diagnostics = []
    for code, rid, spec, heat, group, post in collect_actual_specs():
        options = {'fixef_tol': 1e-10, 'fixef_maxiter': 100000} if tight else {}
        model = pf.feols(spec.formula, data=spec.df.copy(deep=True), vcov={'CRV1': 'kabupaten_full_code'}, **options)
        tidy = model.tidy()
        diagnostics.append(dict(table_code=code, regression_id=rid, formula=spec.formula, retained_terms=list(tidy.index), regressor_count=int(model._k), fixed_effect_levels={str(k): int(v) for k,v in model._k_fe.items()}, df_k=int(model._df_k), projection_tolerance=options.get('fixef_tol', 1e-8)))
        def add(effect, metric, value):
            rows.append(dict(table_code=code, regression_id=rid, coefficient=effect, metric=metric, python_value=float(value)))
        for metric, value in [('N', model._N), ('n_clusters', model._G[0]), ('df_t', model._df_t), ('df_k', model._df_k)]:
            add('__model__', metric, value)
        effects = {'heat': heat}
        if group:
            effects['differential_impact_heat'] = ':'.join(x for x in [group, post, heat] if x)
        for effect, term in effects.items():
            for metric, column in [('b', 'Estimate'), ('se', 'Std. Error'), ('p', 'Pr(>|t|)')]:
                add(effect, metric, tidy.loc[term, column])
        kept = np.array([i for i in range(len(spec.df)) if i not in model._na_index])
        sample = spec.df.iloc[kept][['pidlink', 'wave']].copy()
        sample.insert(0, 'stata_source_row', kept)
        sample.to_csv(OUT / f'{code}_{rid}_python_sample.csv', index=False)
        print(f'{code}/{rid}: N={model._N}, clusters={model._G[0]}, df_k={model._df_k}', flush=True)
    (DELIVERY / 'estimation_diagnostics.json').write_text(json.dumps(dict(pyfixest_version=pf.__version__, python_hash_seed=os.environ.get('PYTHONHASHSEED'), models=diagnostics), indent=2)+'\n')
    reference = pd.DataFrame(rows)
    reference.to_csv(DELIVERY / 'python_reference.csv', index=False)
    return reference


def compare_with_stata(reference):
    stata_path = OUT / 'stata_results.csv'
    keys = ['table_code', 'regression_id', 'coefficient', 'metric']
    if stata_path.exists():
        stata = pd.read_csv(stata_path).rename(columns={'value': 'stata_value'})
        comparison = reference.merge(stata, on=keys, how='outer', validate='one_to_one')
    else:
        comparison = reference.assign(stata_value=np.nan)
    comparison['difference'] = comparison['stata_value'] - comparison['python_value']
    # Tight enough to detect coding differences; accommodates iterative FE projection.
    tolerances = {'N': 0, 'n_clusters': 0, 'df_t': 0, 'df_k': 0, 'b': 1e-7, 'se': 1e-7, 'p': 1e-6}
    comparison['absolute_tolerance'] = comparison.metric.map(tolerances)
    comparison['passed'] = comparison.difference.abs() <= comparison.absolute_tolerance
    comparison.to_csv(DELIVERY / 'python_stata_comparison.csv', index=False)
    samples = []
    for code, rid in reference[['table_code', 'regression_id']].drop_duplicates().itertuples(index=False):
        path = OUT / f'{code}_{rid}_sample.csv'
        py = pd.read_csv(OUT / f'{code}_{rid}_python_sample.csv', dtype={'pidlink': str, 'wave': str})
        if path.exists():
            st = pd.read_csv(path, dtype={'pidlink': str, 'wave': str})
            py_rows = set(py.stata_source_row); st_rows = set(st.stata_source_row)
            joined = py.merge(st, on='stata_source_row', suffixes=('_py','_st'), how='inner')
            identities = ((joined.pidlink_py == joined.pidlink_st) & (joined.wave_py == joined.wave_st)).all()
            samples.append(dict(table_code=code, regression_id=rid, python_N=len(py), stata_N=len(st), python_only=len(py_rows-st_rows), stata_only=len(st_rows-py_rows), identifiers_match=bool(identities)))
        else:
            samples.append(dict(table_code=code, regression_id=rid, python_N=len(py), stata_N=np.nan, python_only=np.nan, stata_only=np.nan, identifiers_match=False))
    sample_comparison = pd.DataFrame(samples)
    sample_comparison.to_csv(DELIVERY / 'sample_comparison.csv', index=False)
    summary = dict(models=len(samples), numerical_rows=len(comparison), numerical_passed=int(comparison.passed.sum()), numerical_unperformed=int(comparison.stata_value.isna().sum()), samples_matched=int(((sample_comparison.python_only == 0) & (sample_comparison.stata_only == 0) & sample_comparison.identifiers_match).sum()), tolerances=tolerances)
    (DELIVERY / 'validation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tight', action='store_true', help='Diagnostic: tighten FE projection without changing scientific specification')
    args = parser.parse_args()
    if args.tight:
        DELIVERY = DELIVERY / 'tight_projection'
    compare_with_stata(estimate_fresh_references(tight=args.tight))
