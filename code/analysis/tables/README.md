# Analysis tables

Each `table_<letter>_*.py` is a standalone script that writes LaTeX (`*.tex`,
`*_body.tex`) and a `*.csv` to `output/tables/`. Run from the repo root, e.g.

```sh
uv run python code/analysis/tables/table_a_headline.py
```

Shared helpers (sample restriction, fixed-effect fitting, LaTeX cells, output
writing) live in `_lettered_common.py`.

## Data dependencies

`data/` is **gitignored** — these inputs are placed locally, not committed. A
reviewer cloning fresh must populate them before the affected tables will run.

**Every table** reads the canonical analysis input:

- `data/generated/30_analysis_table_input.parquet` — built by the data pipeline:
  `uv run python code/data/main.py` (see `code/data/DATA.md`).

**Some tables additionally read raw IFLS sidecars or external reference files**
(everything not listed here needs only the parquet):

| Table | Extra input | Canonical path(s) |
|---|---|---|
| M — palm smallholder mechanism | IFLS `b3a_tk2.dta` (IFLS4 + IFLS5) | `data/raw/IFLS/extracted/IFLS4/hh07/b3a_tk2.dta`, `.../IFLS5/hh14/b3a_tk2.dta` |
| O — SW three stressors | IFLS `b3a_sw.dta` (IFLS4 + IFLS5) | `.../{IFLS4/hh07,IFLS5/hh14}/b3a_sw.dta` |
| P — job-loss slope controls | IFLS `b3a_tk2.dta` (IFLS4 only) | `data/raw/IFLS/extracted/IFLS4/hh07/b3a_tk2.dta` |
| Q — palm-intensity dose-response | BPS provincial palm-area CSV | `data/raw/palm_area_prov_BPS.csv` |
| S — palm placebo | IFLS `b2_ut1.dta` (IFLS4 + IFLS5) | `.../{IFLS4/hh07,IFLS5/hh14}/b2_ut1.dta` |
| T — crop–palm overlap diagnostic | IFLS `b2_ut1.dta` (IFLS4 + IFLS5) | `.../{IFLS4/hh07,IFLS5/hh14}/b2_ut1.dta` |

Raw IFLS folders resolve from `code/data/config.py` (`IFLS4_FOLDER`,
`IFLS5_FOLDER`), i.e. `data/raw/IFLS/extracted/IFLS4/hh07` and `.../IFLS5/hh14`.

### External reference files

- **`data/raw/palm_area_prov_BPS.csv`** (Table Q). Schema:
  `prov_code, prov_name, PR_area_2007, PBN_area_2007, PBS_area_2007, TOT_area_2007,
  PR_area_2014, …, TOT_area_2014` — palm planted hectares by province
  (`TOT` = total; `PR`/`PBN`/`PBS` = smallholder / state estate / private estate).
  Source: BPS / Ditjenbun, *Statistik Perkebunan Indonesia: Kelapa Sawit*. Only
  `TOT_area_2007` / `TOT_area_2014` are used (wave-matched).
- **`data/raw/CMO-Historical-Data-Monthly.xlsx`** — World Bank Commodity Markets
  Outlook (Pink Sheet). Not read by any table at run time; the cross-wave
  commodity price changes annotated in Table S are derived from it and embedded
  as constants. Kept under `data/raw/` for reference.

## Notes for reviewers

- **"Palm farmer"** in these tables = *agricultural household × palm-producing
  province*. IFLS records the crop a household grows (`b2_ut1` `ut07a`/`ut07b`)
  but has **no oil-palm code**, so palm cannot be identified by cultivation —
  hence the region × agriculture proxy. Other crops (rice, maize, rubber, coffee,
  …) *are* identified by actual cultivation. Table T quantifies the resulting
  overlap and shows the crop placebos carry no effect independent of palm.
- Palm-identification tables (Q/R/S/T) use the 7-day mean temperature deviation
  (`tmean_7d_dev`) and kabupaten-clustered standard errors on the IFLS4-baseline
  panel.
