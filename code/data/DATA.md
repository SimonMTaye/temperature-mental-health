# DATA.md - IFLS Data Pipeline

This folder owns the script-first data pipeline for the IFLS mental-health and
temperature project. It turns raw IFLS, geography, pollution, commodity, and
weather inputs into `data/generated/30_analysis_table_input.parquet`, the
canonical input for analysis, tables, and figures.

## Pipeline pattern

Run from the repo root with:

```sh
uv run python code/data/main.py
```

`main.py` first unpacks configured IFLS archives, then runs numbered modules by
dependency layer. Numbering communicates dependency depth:

- `01`/`02`: core IFLS person-wave and geography artifacts.
- `10`-`13`: environmental pulls and exposure source files.
- `20`-`25`: respondent, household, economic, health, commodity, and outcome
  artifacts.
- `30`: final downstream-facing analysis table.

When adding or moving a data step, first identify what upstream artifacts it
needs. Put it in the earliest layer whose dependencies are available, and add it
to `PIPELINE_LAYERS` in `main.py` if it should run in the standard build.

## Data layout

Raw IFLS inputs live under `data/raw/IFLS`; extracted Stata/documentation files
live under `data/raw/IFLS/extracted`. 

Generated pipeline artifacts live under `data/generated`. They are safe to
delete and rebuild. Large generated intermediates belong under the configured
generated-data paths, not in source or resources folders.

GADM and other shared external references are configured in `config.py`. Keep
path constants centralized there rather than scattering repo-relative path math
across scripts.

## Programming conventions

- Use `uv run python ...` for scripts and `uv add ...` for dependencies.
- Keep the pipeline script-first and readable: flat functions, named steps, and
  explicit `main()` entrypoints.
- For derived variables, prefer a helper that takes a `df` and returns a `df`,
  with a short docstring explaining the goal. Wave-specific raw readers should
  be thin wrappers around shared dataframe transforms.
- Add new derived-data helpers as named steps in the merger or builder that
  produces the artifact, so readers can scan one function and see the pipeline.
- Validate generated artifacts at output boundaries with the Pandera schemas in
  `_schemas.py`; do not bury schema checks inside every small transform.
- Avoid overly defensive programming for fixed IFLS layouts. Assume known files
  and columns, and let crashes expose real layout changes.
- Use `_sentinels.py` for IFLS missing/refused/don't-know codes, `_ifls_wave.py`
  for IFLS4/IFLS5 metadata, `_stata.py` for eager Stata reads, and
  `_commodity_prices.py` for shared commodity price lookups.

## Environmental data and GEE

The current pipeline uses Google Earth Engine for ERA5-Land temperature,
MERRA-2 PM2.5, and MODIS AOD pulls. Use the existing `config.py`/environment
setup and never commit API keys, project IDs from private `.env` files, or other
secrets.

Temperature linkage should link survey dates and locations to daily environmental variation, keep lead/lag
windows comparable, and reuse shared scoring or temperature-processing helpers
when the interface fits. 
## Files

- `main.py`: idempotently unpacks configured IFLS archives, then runs ordered
  pipeline modules by dependency layer.
- `config.py`: central path constants for the project, raw IFLS data, generated
  data, output folders, GADM, and GEE settings.
- `01_extract_individuals.py`: builds person-wave IFLS4/IFLS5 interview records
  with dates and normalized admin codes.
- `02_build_geography.py`: builds the kabupaten polygon/centroid lookup from
  BPS and GADM sources.
- `10_fetch_temperature_gee.py`: pulls ERA5-Land daily polygon-mean temperature
  and weather variables.
- `11_fetch_temperature_hourly_gee.py`: pulls ERA5-Land hourly temperature and
  dewpoint for within-day heat analysis.
- `12_fetch_merra_pm25_gee.py`: builds daily kabupaten PM2.5 from MERRA-2
  aerosol components.
- `13_fetch_aod_gee.py`: builds monthly kabupaten MODIS AOD as a haze proxy.
- `20_build_economic_exposures.py`: builds job-loss, asset, benefit-card, and
  palm-price exposure variables.
- `21_build_health_exposures.py`: builds acute health, hospitalization,
  accident, and bereavement shocks.
- `22_build_person_covariates.py`: builds demographic, socioeconomic,
  household, and baseline stressor covariates.
- `23_build_finance_distress.py`: builds household financial-distress and
  economic-shock indicators.
- `24_score_cesd.py`: scores CES-D outcomes and factors for IFLS mental-health
  modules.
- `25_build_commodity_transport_exposures.py`: builds agriculture,
  commodity-region/farmer, and transport-cost exposure measures.
- `30_build_analysis_table_input.py`: merges outcomes, covariates, exposures,
  pollution, and weather into the canonical analysis table.
- `_schemas.py`: Pandera schemas for generated pipeline artifacts.
- `_ifls_wave.py`: IFLS4/IFLS5 metadata for file paths, IDs, interview dates,
  and screening columns.
- `_sentinels.py`: IFLS sentinel-code cleaning helpers.
- `_stata.py`: typed eager Stata reader wrapper.
- `_commodity_prices.py`: hard-coded commodity-price series and decline helpers.

## Anti-patterns
- Do not duplicate this guide's data-pipeline rules in `AGENTS.md`; keep
  `AGENTS.md` as the high-level project guide.
