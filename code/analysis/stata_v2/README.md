# Stata port and validation

From the repository root, after generating the Python exports:

```sh
/Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp -b do code/analysis/stata_v2/reconcile_first.do
/Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp -b do code/analysis/stata_v2/port_tables.do
PYTHONHASHSEED=1 uv run --no-sync python code/analysis/stata_v2/validate_port.py
PYTHONHASHSEED=0 uv run --no-sync python code/analysis/stata_v2/validate_port.py --tight
```

The first command reconciles A/1, B/1, and B/2; the second contains all 29 explicit `use` and `reghdfe` commands. It must finish through `postclose` and the results export. Stata process exit status alone is insufficient: inspect `.cache/stata_port/port.log` for errors. The Python validator uses fresh `pyfixest.feols` calls, bypassing all regression caches. Its default output preserves the Python table engine defaults. `--tight` is a separately labeled numerical diagnostic using `fixef_tol=1e-10`, `fixef_maxiter=100000`, with the same formula, data, covariance, and singleton rules. It does not change the analysis scripts or their existing estimates.

Stata requires `reghdfe`, `ftools`, and `require`. Installed versions used here: reghdfe 6.13.1, ftools 2.50.0, require 1.3.1, StataNow/MP 19.5. `ftools` and `require` were installed into workspace-local `.cache/stata_port/ado`; the do-files add that path for this run only. To reproduce the local dependency setup:

```stata
net set ado ".cache/stata_port/ado"
net install ftools, from("https://raw.githubusercontent.com/sergiocorreia/ftools/master/src/") replace
ssc install require, replace
```

Ensure `.cache/stata_port/ado` exists first. A globally installed compatible dependency also works. In this managed workspace the Python commands used `UV_CACHE_DIR=/private/tmp/ifls-uv-cache MPLCONFIGDIR=/private/tmp/ifls-mpl`.

## Translation decisions

`month*year` inside the installed PyFixest FE parser is a numeric multiplication before factorization. The port absorbs the exported `stata_month_times_year` product. It deliberately preserves this actual specification; an intended calendar month-year FE would require a separately authorized scientific change. `gadm_fullcode` is the community FE identifier; labels in rendered tables are not used to infer its meaning.

All supplied controls, including ethnicity and religion codes, are numeric continuous regressors. Numeric 0/1 regressors and Python booleans retain their meanings; `c.` interaction notation yields products, with baseline 0. The panel models explicitly include only the manually requested five heat/group/post terms and controls; individual and wave FE are absorbed. Complete-case and recursive singleton selection are left to the engines. Kabupaten clusters apply to every model. `dof(clusters)` avoids Stata's additional pairwise FE rank adjustment and follows the Python `k_fixef="nonnested"` convention as closely as possible.

Only heat and differential heat are displayed by the active tables. There are no displayed linear combinations in these active table functions: `lincom` exports each displayed single coefficient with its standard error and unadjusted t p-value. The unused `make_shock_regression_table` helper has a treated-heat combination, but A/B/D do not call it.

## Validation artifacts

`validation/python_reference.csv`, `python_stata_comparison.csv`, `sample_comparison.csv`, and `validation_summary.json` record the default Python comparison. `validation/tight_projection/` records the separate tighter projection diagnostic. Comparison rows identify table, regression ID, coefficient/contrast, metric, Python value, Stata value, and difference. Full respondent-wave sample CSVs and raw Stata results stay in ignored `.cache/stata_port/` because they contain individual identifiers.

Count/cluster/df tolerances are exact; coefficients and SE use absolute tolerance `1e-7`, p-values `1e-6`. These are far below the tables' three-decimal display precision and stringent enough to flag small solver/finite-sample differences. Passing a tolerance is a numerical comparison, not a claim of exact equality. Native default discrepancies remain visible, even when the tight diagnostic explains them.

## Coefficient targets for future multiple testing

A/A2's displayed heat term is the corresponding heat variable in the explicit command. Other targets are:

| Table / ID | Heat coefficient | Differential heat coefficient |
|---|---|---|
| table_b / 1 | `tmean_7d` | `c.palm_farmer_hh_ifls4#c.ifls5#c.tmean_7d` |
| table_b / 2 | `tmean_7d` | `c.palm_farmer_hh_ifls4#c.ifls5#c.tmean_7d` |
| table_b / 3 | `tmean_7d` | `c.palm_price_gap_z#c.ifls5#c.tmean_7d` |
| table_b / 4 | `tmean_7d` | `c.urban_vehicle_hh_ifls4#c.post_subsidy#c.tmean_7d` |
| table_b / 5 | `tmean_7d` | `c.urban_vehicle_transfer__0dc7241a#c.post_subsidy#c.tmean_7d` |
| table_b / 6 | `tmean_7d` | `c.urban_vehicle_transfer__60e2d6a2#c.post_subsidy#c.tmean_7d` |
| table_b / 7 | `tmean_7d` | `c.job_loss_180d#c.tmean_7d` |
| table_b / 8 | `wetbulb_7d` | `c.palm_farmer_hh_ifls4#c.ifls5#c.wetbulb_7d` |
| table_b / 9 | `wetbulb_7d` | `c.palm_farmer_hh_ifls4#c.ifls5#c.wetbulb_7d` |
| table_b / 10 | `wetbulb_7d` | `c.palm_price_gap_z#c.ifls5#c.wetbulb_7d` |
| table_b / 11 | `wetbulb_7d` | `c.urban_vehicle_hh_ifls4#c.post_subsidy#c.wetbulb_7d` |
| table_b / 12 | `wetbulb_7d` | `c.urban_vehicle_transfer__0dc7241a#c.post_subsidy#c.wetbulb_7d` |
| table_b / 13 | `wetbulb_7d` | `c.urban_vehicle_transfer__60e2d6a2#c.post_subsidy#c.wetbulb_7d` |
| table_b / 14 | `wetbulb_7d` | `c.job_loss_180d#c.wetbulb_7d` |
| table_d / 1 | `tmean_7d` | `c.palm_farmer_hh_ifls4#c.ifls5#c.tmean_7d` |
| table_d / 2 | `tmean_7d` | `c.palm_farmer_both_waves#c.ifls5#c.tmean_7d` |
| table_d / 3 | `tmean_7d` | `c.palm_farmer_hh_ifls5#c.ifls5#c.tmean_7d` |

No rwolf2/wyoung adjustments or hypothesis families are implemented. Separate per-regression exports support this numerical port. Future family-level resampling needs jointly aligned data and shared bootstrap draws; independently bootstrapping separately loaded datasets is not a valid joint adjustment.

## Completed validation results

All 29 models ran in Stata, following the initial A/1, B/1, B/2 reconciliation. Every respondent-wave estimation sample matched exactly, as did all observation counts, cluster counts, and t-test degrees of freedom.

The reproducible default Python run (`PYTHONHASHSEED=1`, PyFixest 0.50.1, projection tolerance `1e-8`) passes 212/254 metric rows. Maximum absolute differences: coefficient 2.51e-07, SE 1.12e-06, p-value 4.54e-06; seven regressions have Python `df_k` one above Stata. The default engine numerically retains the fully absorbed `ifls5` regressor in B/1, B/3, B/8, B/10 and D/1–3, counting an extra parameter in its CRV1 finite-sample denominator. `estimation_diagnostics.json` records all retained terms and FE counts. FE terms are assembled through a Python set, so hash order can affect projection and this borderline collinearity detection. The scientific specification was left unchanged.

The separate diagnostic (`PYTHONHASHSEED=0`, projection tolerance `1e-10`, maximum 100000 iterations) passes **254/254 metrics** and all **29/29 exact samples**. Maximum absolute differences: coefficient 6.46e-12, SE 1.32e-10, p-value 1.56e-09; all `df_k` values match. Tightening projection removes the absorbed wave regressor and explains the native default discrepancies. This establishes equivalent regression translation at tighter numerical precision; exact parity with every default/cached Python result is not claimed. Existing Python table settings and caches were not modified.
