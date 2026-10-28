# Stata port and validation

Explicit Stata (`reghdfe`) port of the Python tables A, A2 (sleep), B, and D, with a
numerical comparison against fresh `pyfixest` fits. Three steps, all from the repository root:

```sh
uv run python code/analysis/stata_v2/export_port_data.py
/Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp -b do code/analysis/stata_v2/port_tables.do
PYTHONHASHSEED=1 uv run python code/analysis/stata_v2/validate_port.py
PYTHONHASHSEED=0 uv run python code/analysis/stata_v2/validate_port.py --tight
```

Stata's exit status is unreliable in batch mode: check `.cache/stata_port/port.log` for
`r(` error codes (`grep -n "r([0-9]*);" .cache/stata_port/port.log` should print nothing).

## Single dataset export

`export_port_data.py` calls `library.port_helper.save_data(build_palm_df())`, which writes

- `data/generated/stata_port/analysis_port.dta` (all 60,419 respondent-waves, all columns), and
- `data/generated/stata_port/analysis_port_mapping.json` (original-to-Stata column names,
  identifier alias dictionaries, row count).

`build_palm_df()` (in `library/specs.py`) is `analysis_df` plus the alternative palm-farmer
definitions used by table D; it has the same rows as `analysis_df` and is a column superset, so
every ported regression can be run from this one file. Table B's wave-5-only models use
`wave5_df = analysis_df[analysis_df.ifls5]` in Python and `if ifls5 == 1` in Stata (B IDs 4-6
and 11-13). Nothing else is restricted before estimation: complete-case and recursive
singleton dropping are left to the engines, as in Python.

Export conventions (unchanged from the per-regression version):

- Names over 32 characters, or containing non-identifier characters, become the first 23
  characters plus `_` and 8 SHA-256 hex digits of the original name (for example
  `urban_vehicle_transfer_nonrecipient_ifls4` -> `urban_vehicle_transfer__0dc7241a`); the
  mapping sidecar is authoritative. Collisions raise.
- Nullable numeric and boolean columns become doubles with ordinary missing values;
  categoricals keep their values; missing strings become empty strings; datetimes are `%tc`.
- Helper columns: `stata_source_row` (0-based row position), `stata_pidlink` and
  `stata_gadm_fullcode` (1-based sorted numeric aliases of the string identifiers, used only
  as FE), and `stata_month_times_year` (`month * year`, see below).
- Set `OUTPUT_DATA = False` in `library/port_helper.py` to disable writes.

## Do-file

`port_tables.do` loads the dataset once, runs the 29 regressions with a comment header per
model giving the regression ID and the resolved Python formula, stores each with `eststo`, and
writes one `esttab` table per panel to `output/tables/stata_port/` (ignored by git):

| File | Models | Displayed terms |
|---|---|---|
| `table_a.{tex,csv}` | `table_a_1`..`_6` | heat |
| `table_a2.{tex,csv}` | `table_a2_1`..`_6` | heat |
| `table_b_temperature.{tex,csv}` | `table_b_1`..`_7` | heat, differential heat |
| `table_b_wetbulb.{tex,csv}` | `table_b_8`..`_14` | heat, differential heat |
| `table_d.{tex,csv}` | `table_d_1`..`_3` | heat, differential heat |

The `.tex` uses `se star b(%9.4f) se(%9.4f) stats(N N_clust)`; the `.csv` is the same
`keep()` list written with `cells("b se")` at `%20.15g` for the validator.

Every model uses `vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)` and absorbs
`stata_month_times_year stata_gadm_fullcode` (panel models add `stata_pidlink` and the post
indicator). Stata needs `reghdfe` and `estout` (installed globally here) plus `ftools` and
`require`, which live in the workspace-local, git-ignored `.cache/stata_port/ado` that the
do-file adds to the adopath. Recreate them with:

```sh
mkdir -p .cache/stata_port/ado
```

```stata
net set ado ".cache/stata_port/ado"
net install ftools, from("https://raw.githubusercontent.com/sergiocorreia/ftools/master/src/") replace
ssc install require, replace
```

Versions used: StataNow/MP 19.5, reghdfe 6.13.1, ftools 2.50.0, require 1.3.1, estout 3.x.

## Translation decisions

- `month*year` in the pyfixest FE formula is the numeric product, categorized. For this data
  (years 2007/08/14/15, months 1-12) the product is a bijection with (month, year), so
  absorbing `stata_month_times_year` is a month-by-year FE. `gadm_fullcode` is the community FE.
- All controls, including ethnicity and religion codes, enter as numeric regressors, exactly
  as in Python. `c.` interaction notation yields products; `##` expands the full interaction
  set that `*` produces in pyfixest. Panel models include only the five requested
  heat/group/post terms plus controls.
- `dof(clusters)` avoids reghdfe's pairwise FE rank adjustment and follows pyfixest's
  `k_fixef="nonnested"` convention.

## Validation

`validate_port.py` rebuilds the 29 specs from the table scripts, fits each fresh with
`pyfixest.feols` on `build_palm_df()` (restricted to `ifls5` for the wave-5 models) with
`vcov={"CRV1": "kabupaten_full_code"}`, parses the esttab CSVs, and compares `b` and `se` of
the displayed terms plus `N` and `N_clust`. Tolerances: `N`, `N_clust` exact; `b`, `se` absolute
`1e-7`. It writes `python_stata_comparison.csv` and `validation_summary.json` under
`.cache/stata_port/validation/` (`tight_projection/` for `--tight`). Coefficient names in the
CSV for the differential-heat term are `c.<group>#c.<post>#c.<heat>` (or `c.<group>#c.<heat>`
for job loss), using the exported group name from the mapping sidecar.

Results on the current data (29/29 models, N and N_clust exact everywhere):

- Default pyfixest projection (`PYTHONHASHSEED=1`, `fixef_tol=1e-8`): 127/150 rows within
  tolerance; max |diff| coefficient 2.51e-07, SE 1.12e-06. The excess is iterative-projection
  noise: with default settings pyfixest retains the fully absorbed `ifls5` regressor in
  B/1, B/3, B/8, B/10 and D/1-3 (one extra parameter in the CRV1 small-sample factor).
- `--tight` (`fixef_tol=1e-10`, `fixef_maxiter=100000`): 150/150 rows; max |diff| coefficient
  6.46e-12, SE 1.32e-10. This is the equivalence check; the analysis scripts and their
  cached estimates are unchanged.

No multiple-testing adjustment (rwolf2/wyoung) is implemented. Because all models now sit on a
single dataset, jointly aligned resampling is possible in a later stage.

## Multiple hypothesis testing

Family-wise and FDR-adjusted inference for the two hypothesis families the paper reports,
all run on the same `analysis_port.dta`:

| Family | Hypotheses | Coefficient tested |
|---|---|---|
| A (6) | `table_a_1`..`_6` | heat measure (`tmean_7d`, `hot30_7d`, `wetbulb_7d`, `tmean_c`, `tmax_c`, `wetbulb_c`) in the CES-D regression |
| B (14) | `table_b_1`..`_14` | differential-heat interaction (`shock x post x heat`, or `job_loss_180d x heat`); 1-7 `tmean_7d`, 8-14 `wetbulb_7d` |

Table A2 (sleep) and table D are excluded.

**Cluster.** All MHT models cluster on `kabupaten_full_code_ifls4`: each person's IFLS4
kabupaten carried to both waves via `pidlink` (IFLS5-only entrants keep their own
kabupaten). `export_port_data.py` adds this column and `interview_date` (calendar day)
to the export; nothing else in the export changed. Reason: 9.1% of panel individuals change
kabupaten between IFLS4 and IFLS5, so a cluster bootstrap on the contemporaneous kabupaten
would split a person's two observations across independently drawn clusters and
mis-specify the `pidlink`-FE models (B/2, B/9) under resampling. Point estimates are
unchanged; p-values move materially only for B/2 and B/9 (SE 0.0307 -> 0.0222 and
0.0323 -> 0.0233). The full stage-1 vs. baseline-cluster comparison is in
`.cache/stata_port/mht/check_cluster_comparison.csv`.

**Files** (all in this directory; run from the repository root after `export_port_data.py`):

| File | Role |
|---|---|
| `mht_models.do` | Defines the 20 models once (globals `mht_cmd1`..`mht_cmd20`, `mht_target*`, `mht_fam*`), the slim loader `mht_load`, and `mht_build_interactions`. The `reghdfe` lines are those of `port_tables.do` with the new cluster and with `c.a##c.b##c.c` written as explicit product variables (`rwolf2` reads its targets as a plain varlist); coefficients match stage 1 to 15 digits. Included by every other file so no regression is copied. |
| `mht_uncorrected.do` | Fits the 20 models, writes `.cache/stata_port/mht/uncorrected.csv` (b, se, t, clustered p, N, N_clust). |
| `mht_rwolf.do A|B reps` | Romano-Wolf step-down via `rwolf2` (SSC, installed in `.cache/stata_port/ado`): cluster bootstrap `bsample, cluster(kabupaten_full_code_ifls4) idcluster(bs_cluster)`, centred studentized statistics, max-|t| step-down, plus Holm. Writes `rwolf_<family>.csv`. `vce()` is the last option of every command so `rwolf2`'s `idcluster()` regex rewrite only touches it. |
| `mht_qvalues.do` | Anderson (2008) sharpened two-stage BKY q-values (`fdr_sharpened_qvalues.do` algorithm, step 0.001), applied family by family to `uncorrected.csv`. Writes `qvalues.csv`. |
| `mht_ri.do A|B perms [chunk]` | Randomisation inference: within each community x wave stratum the community x `interview_date` blocks' weather vectors (`$mht_weather`: `tmean_c tmax_c tmin_c heat_idx_c rh_pct tmean_7d tmin_7d tmean_30d wetbulb_c wetbulb_7d wetbulb_30d hot30_7d heatwave_7d`) are permuted among the stratum's dates, all columns together; every model in the family is refitted (same cluster, same `if`) and |t| of the target recorded. Per-model p compares the actual |t| with that model's permutation distribution; the family-wise p compares it with the distribution of max |t| over the family (single-step Westfall-Young). Without `chunk`: writes `ri_<family>_perms.dta`, `ri_<family>_checks.csv`, `ri_<family>.csv`. With `chunk`: see "Chunked RI runs" below. |
| `mht_ri_combine.do A|B` | Appends the chunked `ri_<family>_perms_<chunk>.dta` files (plus `ri_<family>_perms.dta` if present) into `ri_<family>_allperms.dta`, refits the actual statistics once (tight tolerance) and writes `ri_<family>.csv` in the single-run format, with `perms` = total appended permutations. The p-value code (`mht_ri_pvalues`) and the actual-statistic code (`mht_ri_actual`) live in `mht_models.do` and are shared with `mht_ri.do`. |
| `mht_tables.do` | Refits the 20 models with `eststo`, `estadd`s the adjusted p-values from the CSVs, and writes `output/tables/stata_port/table_mht_{rwolf,qvalues,ri}.tex` (Family A panel, 6 columns; Family B panel, 14 columns; coefficient rows renamed `heat` / `diff_heat`). |

Exact commands (seed 20261009 everywhere; one `reghdfe` fit is about 0.6 s, so the long jobs
are best run with `nohup ... &` and polled through their logs in `.cache/stata_port/mht/`):

```sh
S=/Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp
$S -b do code/analysis/stata_v2/mht_uncorrected.do
$S -b do code/analysis/stata_v2/mht_qvalues.do
$S -b do code/analysis/stata_v2/mht_rwolf.do A 500
$S -b do code/analysis/stata_v2/mht_rwolf.do B 300
$S -b do code/analysis/stata_v2/mht_ri.do A 500
$S -b do code/analysis/stata_v2/mht_ri.do B 500      # or the chunked recipe below
$S -b do code/analysis/stata_v2/mht_tables.do
```

**Chunked RI runs.** Family B refits 14 models per permutation, so 500 permutations in one
process take several hours. `mht_ri.do <family> <perms> <chunk>` runs `perms` permutations
as chunk number `chunk` and stops before the p-values: the seed is `20261009 + chunk`
(base seed plus the chunk number, so chunks are reproducible and draw disjoint random
streams), and the outputs carry the chunk suffix (`ri_<family>_perms_<chunk>.dta`,
`ri_<family>_checks_<chunk>.csv`, `ri_<family>_<chunk>.log`). Run the chunks in parallel
(each Stata process uses about two cores), then combine. Stata's batch mode writes
`<dofile>.log` in the working directory for every process, so launch each chunk through a
one-line wrapper do-file with a distinct name rather than calling `mht_ri.do` directly:

```sh
S=/Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp
mkdir -p .cache/stata_port/mht/jobs
for k in 1 2 3 4; do
  echo "do code/analysis/stata_v2/mht_ri.do B 125 $k" > .cache/stata_port/mht/jobs/ri_B_chunk$k.do
  nohup $S -b do .cache/stata_port/mht/jobs/ri_B_chunk$k.do > /dev/null 2>&1 &
done
# wait until `pgrep -f ri_B_chunk` is empty and `grep -n "r([0-9]*);" .cache/stata_port/mht/ri_B_?.log` prints nothing, then
$S -b do code/analysis/stata_v2/mht_ri_combine.do B
```

`ri_B.csv` then reports `perms` = 500, `seed` = 20261009 (the base seed; chunk k used
20261009 + k) and `secs` = the sum of the chunks' permutation-loop times. The combine step
appends every `ri_<family>_perms_*.dta` it finds, so delete chunk files from an aborted or
superseded run before combining.

**Tolerance.** The permuted refits use `$mht_perm_opts` (`reghdfe`'s default
`tolerance(1e-8)`); the actual estimates reported in `ri_<family>.csv` (b, se, t, p_unadj)
use `$mht_opts` with `tolerance(1e-12)` and still match the port to 15 digits. The
permutation distribution is only used to count exceedances of the actual |t|, so the
trailing digits of each permuted t* are irrelevant, and the HDFE iteration dominates the
per-fit cost: 14 Family B fits take about 54 s at 1e-12 and 11 s at the default (one
permutation of Family B is about 10 s, so 125 permutations per chunk is roughly 25 minutes
with four chunks in parallel).

Install `rwolf2` into the workspace ado directory with `net set ado ".cache/stata_port/ado"`
then `ssc install rwolf2, replace` (version 1.0.0 used).
