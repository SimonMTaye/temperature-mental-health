# Python export handoff

Exports contain the final `spec.df`, including Python derived variables and restrictions, before engine complete-case or singleton handling. All original columns and rows are preserved. The sleep specifications retain both waves in the input; missing outcomes select the estimation sample. D uses the merged `palm_df`.

Run from the repository root (no raw-data pipeline):

```sh
mkdir -p output/tables
uv run python code/analysis/tables_v2/table_a_temperature_effects.py
uv run python code/analysis/tables_v2/table_b_temperature_and_shock_effects.py
uv run python code/analysis/tables_v2/table_d_palm_farmer_alt_definitions.py
```

Set `OUTPUT_DATA = False` in `code/library/port_helper.py` to disable writes. Calls precede cache lookup. B exports only inside its own table builder, so other users of `regression_runner` do not export.

Files live in `data/generated/stata_port/` and are generated, uncommitted inputs. Each file has a `{table}_{id}_mapping.json` sidecar with the complete original/export name mapping, row count, and numeric identifier dictionaries. Names over 32 characters use their first 23 characters plus `_` and 8 SHA-256 hex digits; collisions raise an error. No silent Stata writer renaming occurs.

Nullable numeric/boolean values become Stata numeric doubles with ordinary numeric missing values. Existing floats keep precision. Strings retain values; missing strings become Stata empty strings (Stata string missing). Categorical values retain their labels, not category codes. Datetimes use Stata `%tc` milliseconds. `pidlink`, `wave`, and all original identifiers remain present. `stata_source_row` is the zero-based row position in the exported final dataframe, enabling direct estimation-sample reconciliation. `stata_pidlink` and `stata_gadm_fullcode` provide 1-based sorted numeric identifier aliases, with missing identifiers left missing. `stata_month_times_year` is the numeric product `month * year`, preserving missingness: pyfixest's fixed-effect syntax categorizes that product; it is not a month-by-year interaction. Regression covariates are never replaced by identifier codes.

Every model uses the actual covariance call `vcov={"CRV1": "kabupaten_full_code"}` through the caching helper (B passes the same setting explicitly). Existing cache keys omit covariance, so port validation must use fresh estimates.

| Code / ID | Source script | Table / panel / column | File | Resolved Python formula | Covariance |
|---|---|---|---|---|---|
| table_a / 1 | `code/analysis/tables_v2/table_a_temperature_effects.py` | CES-D / 1: tmean_7d | `table_a_1.dta` | `cesd_z ~ tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_a / 2 | `code/analysis/tables_v2/table_a_temperature_effects.py` | CES-D / 2: hot30_7d | `table_a_2.dta` | `cesd_z ~ hot30_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_a / 3 | `code/analysis/tables_v2/table_a_temperature_effects.py` | CES-D / 3: wetbulb_7d | `table_a_3.dta` | `cesd_z ~ wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_a / 4 | `code/analysis/tables_v2/table_a_temperature_effects.py` | CES-D / 4: tmean_c | `table_a_4.dta` | `cesd_z ~ tmean_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_a / 5 | `code/analysis/tables_v2/table_a_temperature_effects.py` | CES-D / 5: tmax_c | `table_a_5.dta` | `cesd_z ~ tmax_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_a / 6 | `code/analysis/tables_v2/table_a_temperature_effects.py` | CES-D / 6: wetbulb_c | `table_a_6.dta` | `cesd_z ~ wetbulb_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_a2 / 1 | `code/analysis/tables_v2/table_a_temperature_effects.py` | Sleep / 1: tmean_7d | `table_a2_1.dta` | `sleep_dur_h ~ tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_a2 / 2 | `code/analysis/tables_v2/table_a_temperature_effects.py` | Sleep / 2: hot30_7d | `table_a2_2.dta` | `sleep_dur_h ~ hot30_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_a2 / 3 | `code/analysis/tables_v2/table_a_temperature_effects.py` | Sleep / 3: wetbulb_7d | `table_a2_3.dta` | `sleep_dur_h ~ wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_a2 / 4 | `code/analysis/tables_v2/table_a_temperature_effects.py` | Sleep / 4: tmean_c | `table_a2_4.dta` | `sleep_dur_h ~ tmean_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_a2 / 5 | `code/analysis/tables_v2/table_a_temperature_effects.py` | Sleep / 5: tmax_c | `table_a2_5.dta` | `sleep_dur_h ~ tmax_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_a2 / 6 | `code/analysis/tables_v2/table_a_temperature_effects.py` | Sleep / 6: wetbulb_c | `table_a2_6.dta` | `sleep_dur_h ~ wetbulb_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_b / 1 | `code/analysis/tables_v2/table_b_temperature_and_shock_effects.py` | Temperature / 1: Palm Shock | `table_b_1.dta` | `cesd_z ~ palm_farmer_hh_ifls4 * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_b / 2 | `code/analysis/tables_v2/table_b_temperature_and_shock_effects.py` | Temperature / 2: \shortstack{Palm Shocks\\Panel} | `table_b_2.dta` | `cesd_z ~ tmean_7d + palm_farmer_hh_ifls4:ifls5:tmean_7d + palm_farmer_hh_ifls4:tmean_7d + ifls5:tmean_7d + palm_farmer_hh_ifls4:ifls5 + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode+pidlink+ifls5` | CRV1, kabupaten_full_code |
| table_b / 3 | `code/analysis/tables_v2/table_b_temperature_and_shock_effects.py` | Temperature / 3: \shortstack{Palm Shock\\Price Drop} | `table_b_3.dta` | `cesd_z ~ palm_price_gap_z * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_b / 4 | `code/analysis/tables_v2/table_b_temperature_and_shock_effects.py` | Temperature / 4: Urban Vehicle Owners | `table_b_4.dta` | `cesd_z ~ urban_vehicle_hh_ifls4 * post_subsidy * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_b / 5 | `code/analysis/tables_v2/table_b_temperature_and_shock_effects.py` | Temperature / 5: \shortstack{Urban Vehicle Owners\\No Cash Transfer} | `table_b_5.dta` | `cesd_z ~ urban_vehicle_transfer_nonrecipient_ifls4 * post_subsidy * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_b / 6 | `code/analysis/tables_v2/table_b_temperature_and_shock_effects.py` | Temperature / 6: \shortstack{Urban Vehicle Owners\\Cash Transfer} | `table_b_6.dta` | `cesd_z ~ urban_vehicle_transfer_recipient_ifls4 * post_subsidy * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_b / 7 | `code/analysis/tables_v2/table_b_temperature_and_shock_effects.py` | Temperature / 7: Job Loss | `table_b_7.dta` | `cesd_z ~ job_loss_180d * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_b / 8 | `code/analysis/tables_v2/table_b_temperature_and_shock_effects.py` | Wet-bulb / 1: Palm Shock | `table_b_8.dta` | `cesd_z ~ palm_farmer_hh_ifls4 * ifls5 * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_b / 9 | `code/analysis/tables_v2/table_b_temperature_and_shock_effects.py` | Wet-bulb / 2: \shortstack{Palm Shocks\\Panel} | `table_b_9.dta` | `cesd_z ~ wetbulb_7d + palm_farmer_hh_ifls4:ifls5:wetbulb_7d + palm_farmer_hh_ifls4:wetbulb_7d + ifls5:wetbulb_7d + palm_farmer_hh_ifls4:ifls5 + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode+pidlink+ifls5` | CRV1, kabupaten_full_code |
| table_b / 10 | `code/analysis/tables_v2/table_b_temperature_and_shock_effects.py` | Wet-bulb / 3: \shortstack{Palm Shock\\Price Drop} | `table_b_10.dta` | `cesd_z ~ palm_price_gap_z * ifls5 * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_b / 11 | `code/analysis/tables_v2/table_b_temperature_and_shock_effects.py` | Wet-bulb / 4: Urban Vehicle Owners | `table_b_11.dta` | `cesd_z ~ urban_vehicle_hh_ifls4 * post_subsidy * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_b / 12 | `code/analysis/tables_v2/table_b_temperature_and_shock_effects.py` | Wet-bulb / 5: \shortstack{Urban Vehicle Owners\\No Cash Transfer} | `table_b_12.dta` | `cesd_z ~ urban_vehicle_transfer_nonrecipient_ifls4 * post_subsidy * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_b / 13 | `code/analysis/tables_v2/table_b_temperature_and_shock_effects.py` | Wet-bulb / 6: \shortstack{Urban Vehicle Owners\\Cash Transfer} | `table_b_13.dta` | `cesd_z ~ urban_vehicle_transfer_recipient_ifls4 * post_subsidy * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_b / 14 | `code/analysis/tables_v2/table_b_temperature_and_shock_effects.py` | Wet-bulb / 7: Job Loss | `table_b_14.dta` | `cesd_z ~ job_loss_180d * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_d / 1 | `code/analysis/tables_v2/table_d_palm_farmer_alt_definitions.py` | Alternative definition / 1: palm_farmer_hh_ifls4 | `table_d_1.dta` | `cesd_z ~ palm_farmer_hh_ifls4 * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_d / 2 | `code/analysis/tables_v2/table_d_palm_farmer_alt_definitions.py` | Alternative definition / 2: palm_farmer_both_waves | `table_d_2.dta` | `cesd_z ~ palm_farmer_both_waves * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |
| table_d / 3 | `code/analysis/tables_v2/table_d_palm_farmer_alt_definitions.py` | Alternative definition / 3: palm_farmer_hh_ifls5 | `table_d_3.dta` | `cesd_z ~ palm_farmer_hh_ifls5 * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable &#124; month*year+gadm_fullcode` | CRV1, kabupaten_full_code |

## Exported name changes

Names not listed retain their Python names. Complete per-file dictionaries are in the generated sidecars.

| Python name | Stata name |
|---|---|
| `cash_transfer_blt_blsm_card_ifls4` | `cash_transfer_blt_blsm__aeabd765` |
| `cash_transfer_bpjs_accident_ifls4` | `cash_transfer_bpjs_acci_9597df2f` |
| `cash_transfer_bpjs_retirement_ifls4` | `cash_transfer_bpjs_reti_44f3ce9e` |
| `cash_transfer_child_welfare_ifls4` | `cash_transfer_child_wel_7b840579` |
| `cash_transfer_poverty_certificate` | `cash_transfer_poverty_c_6a853405` |
| `cash_transfer_poverty_certificate_ifls4` | `cash_transfer_poverty_c_7ae42082` |
| `cash_transfer_social_security_card` | `cash_transfer_social_se_48696f60` |
| `cash_transfer_social_security_card_ifls4` | `cash_transfer_social_se_60714237` |
| `cash_transfer_troubled_youth_ifls4` | `cash_transfer_troubled__d4ec101f` |
| `expenditure_food_total_mo_nominal_usd` | `expenditure_food_total__74a27366` |
| `expenditure_food_total_mo_real_usd` | `expenditure_food_total__95591399` |
| `expenditure_nonfood_children_education_mo` | `expenditure_nonfood_chi_b5e0afe3` |
| `expenditure_nonfood_fuel_mo_nominal_usd` | `expenditure_nonfood_fue_06aca34a` |
| `expenditure_nonfood_fuel_mo_real_usd` | `expenditure_nonfood_fue_27006ed8` |
| `expenditure_nonfood_total_mo_nominal_usd` | `expenditure_nonfood_tot_fb03fcb8` |
| `expenditure_nonfood_total_mo_real_usd` | `expenditure_nonfood_tot_e06291e1` |
| `expenditure_nonfood_vehicle_fuel_mo` | `expenditure_nonfood_veh_75816893` |
| `expenditure_nonfood_vehicle_fuel_mo_nominal_usd` | `expenditure_nonfood_veh_acd5372d` |
| `expenditure_transport_fuel_total_mo` | `expenditure_transport_f_dfed0835` |
| `expenditure_transport_fuel_total_mo_nominal_usd` | `expenditure_transport_f_4acd45a6` |
| `expenditure_transport_fuel_total_mo_real_usd` | `expenditure_transport_f_a022141d` |
| `hh_nonlabor_income_mo_nominal_usd` | `hh_nonlabor_income_mo_n_7dcaf1d4` |
| `job_earnings_individual_nominal_usd` | `job_earnings_individual_54b617ab` |
| `travel_districtcapital_distance_z` | `travel_districtcapital__f3fc4642` |
| `travel_provincecapital_distance_z` | `travel_provincecapital__c73ccd75` |
| `urban_vehicle_transfer_nonrecipient_ifls4` | `urban_vehicle_transfer__0dc7241a` |
| `urban_vehicle_transfer_recipient_ifls4` | `urban_vehicle_transfer__60e2d6a2` |

## Verification

All three targeted entrypoints ran, generating 29 readable Stata 118 datasets. All columns were checked against each actual source dataframe, including exact numeric values and missingness, string values/missing representation, dates, identifiers and row counts. A dataframe digest before/after each export confirmed no source mutation. Setting `OUTPUT_DATA = False` around each save left all generated file timestamps and the dataframe digest unchanged. No compatibility issues remain for these observed inputs. A separate nullable-boolean and numeric/string-categorical smoke check passed, including the disabled early return with a null dataset. Numeric Stata regression replication is a separate task.

In this managed workspace, `data/` points outside the writable root; writing exports requires shell approval. The verification run used `MPLCONFIGDIR=/private/tmp/ifls-mpl UV_CACHE_DIR=/private/tmp/ifls-uv-cache uv run --no-sync python /private/tmp/verify_ifls_exports.py` to invoke all three `make_table()` entrypoints with round-trip assertions. The temporary verification harness is not required for ordinary exports.
