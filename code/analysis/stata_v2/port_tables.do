version 19
clear all
set more off
* Explicit Stata port of tables A, A2, B, D. Run from the repository root:
*   uv run python code/analysis/stata_v2/export_port_data.py
*   /Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp -b do code/analysis/stata_v2/port_tables.do
* All variable construction happens in Python; the single export is build_palm_df()
* (a column superset of analysis_df). Wave-5-only models restrict with `if ifls5 == 1`.
capture mkdir ".cache"
capture mkdir ".cache/stata_port"
capture mkdir "output/tables"
capture mkdir "output/tables/stata_port"
log using ".cache/stata_port/port.log", text replace
adopath ++ ".cache/stata_port/ado"
which reghdfe
which ftools
which esttab

* PyFixest: month*year categorizes the numeric product, not calendar month-year
* (for this data the product is a bijection with (month, year), so it is equivalent).
* Numeric covariates (including ethnicity/religion) retain continuous coding.
* Singleton observations are recursively dropped in both engines.
* dof(clusters) mirrors PyFixest nonnested FE counts without pairwise rank adjustment.
* Identification assumes heat is conditionally unrelated to unobserved distress,
* given community FE, FE of numeric month*year, and included controls.
global controls "age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable"
global fe "stata_month_times_year stata_gadm_fullcode"
global opts "vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)"

use "data/generated/stata_port/analysis_port.dta", clear
eststo clear

**** table_a: cesd_z on each heat measure (IDs 1-6)
* code/analysis/tables_v2/table_a_temperature_effects.py; table_a ID 1; CES-D / 1: tmean_7d
* Python: cesd_z ~ tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z tmean_7d $controls, absorb($fe) $opts
eststo table_a_1

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a ID 2; CES-D / 2: hot30_7d
* Python: cesd_z ~ hot30_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z hot30_7d $controls, absorb($fe) $opts
eststo table_a_2

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a ID 3; CES-D / 3: wetbulb_7d
* Python: cesd_z ~ wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z wetbulb_7d $controls, absorb($fe) $opts
eststo table_a_3

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a ID 4; CES-D / 4: tmean_c
* Python: cesd_z ~ tmean_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z tmean_c $controls, absorb($fe) $opts
eststo table_a_4

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a ID 5; CES-D / 5: tmax_c
* Python: cesd_z ~ tmax_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z tmax_c $controls, absorb($fe) $opts
eststo table_a_5

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a ID 6; CES-D / 6: wetbulb_c
* Python: cesd_z ~ wetbulb_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z wetbulb_c $controls, absorb($fe) $opts
eststo table_a_6

**** table_a2: sleep_dur_h on each heat measure (IDs 1-6)
* code/analysis/tables_v2/table_a_temperature_effects.py; table_a2 ID 1; Sleep / 1: tmean_7d
* Python: sleep_dur_h ~ tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe sleep_dur_h tmean_7d $controls, absorb($fe) $opts
eststo table_a2_1

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a2 ID 2; Sleep / 2: hot30_7d
* Python: sleep_dur_h ~ hot30_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe sleep_dur_h hot30_7d $controls, absorb($fe) $opts
eststo table_a2_2

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a2 ID 3; Sleep / 3: wetbulb_7d
* Python: sleep_dur_h ~ wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe sleep_dur_h wetbulb_7d $controls, absorb($fe) $opts
eststo table_a2_3

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a2 ID 4; Sleep / 4: tmean_c
* Python: sleep_dur_h ~ tmean_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe sleep_dur_h tmean_c $controls, absorb($fe) $opts
eststo table_a2_4

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a2 ID 5; Sleep / 5: tmax_c
* Python: sleep_dur_h ~ tmax_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe sleep_dur_h tmax_c $controls, absorb($fe) $opts
eststo table_a2_5

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a2 ID 6; Sleep / 6: wetbulb_c
* Python: sleep_dur_h ~ wetbulb_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe sleep_dur_h wetbulb_c $controls, absorb($fe) $opts
eststo table_a2_6

**** table_b temperature panel: tmean_7d (IDs 1-7)
* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 1; Temperature / 1: Palm Shock
* Python: cesd_z ~ palm_farmer_hh_ifls4 * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z c.palm_farmer_hh_ifls4##c.ifls5##c.tmean_7d $controls, absorb($fe) $opts
eststo table_b_1

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 2; Temperature / 2: \shortstack{Palm Shocks\\Panel}
* Python: cesd_z ~ tmean_7d + palm_farmer_hh_ifls4:ifls5:tmean_7d + palm_farmer_hh_ifls4:tmean_7d + ifls5:tmean_7d + palm_farmer_hh_ifls4:ifls5 + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode+pidlink+ifls5
* Individual FE additionally identify within-person changes; wave FE are absorbed.
reghdfe cesd_z c.tmean_7d c.palm_farmer_hh_ifls4#c.ifls5#c.tmean_7d c.palm_farmer_hh_ifls4#c.tmean_7d c.ifls5#c.tmean_7d c.palm_farmer_hh_ifls4#c.ifls5 $controls, absorb($fe stata_pidlink ifls5) $opts
eststo table_b_2

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 3; Temperature / 3: \shortstack{Palm Shock\\Price Drop}
* Python: cesd_z ~ palm_price_gap_z * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z c.palm_price_gap_z##c.ifls5##c.tmean_7d $controls, absorb($fe) $opts
eststo table_b_3

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 4; Temperature / 4: Urban Vehicle Owners
* Python: cesd_z ~ urban_vehicle_hh_ifls4 * post_subsidy * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
* Python df: wave5_df -> `if ifls5 == 1`
reghdfe cesd_z c.urban_vehicle_hh_ifls4##c.post_subsidy##c.tmean_7d $controls if ifls5 == 1, absorb($fe) $opts
eststo table_b_4

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 5; Temperature / 5: \shortstack{Urban Vehicle Owners\\No Cash Transfer}
* Python: cesd_z ~ urban_vehicle_transfer_nonrecipient_ifls4 * post_subsidy * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
* Python df: wave5_df -> `if ifls5 == 1`
reghdfe cesd_z c.urban_vehicle_transfer__0dc7241a##c.post_subsidy##c.tmean_7d $controls if ifls5 == 1, absorb($fe) $opts
eststo table_b_5

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 6; Temperature / 6: \shortstack{Urban Vehicle Owners\\Cash Transfer}
* Python: cesd_z ~ urban_vehicle_transfer_recipient_ifls4 * post_subsidy * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
* Python df: wave5_df -> `if ifls5 == 1`
reghdfe cesd_z c.urban_vehicle_transfer__60e2d6a2##c.post_subsidy##c.tmean_7d $controls if ifls5 == 1, absorb($fe) $opts
eststo table_b_6

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 7; Temperature / 7: Job Loss
* Python: cesd_z ~ job_loss_180d * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z c.job_loss_180d##c.tmean_7d $controls, absorb($fe) $opts
eststo table_b_7

**** table_b wet-bulb panel: wetbulb_7d (IDs 8-14)
* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 8; Wet-bulb / 1: Palm Shock
* Python: cesd_z ~ palm_farmer_hh_ifls4 * ifls5 * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z c.palm_farmer_hh_ifls4##c.ifls5##c.wetbulb_7d $controls, absorb($fe) $opts
eststo table_b_8

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 9; Wet-bulb / 2: \shortstack{Palm Shocks\\Panel}
* Python: cesd_z ~ wetbulb_7d + palm_farmer_hh_ifls4:ifls5:wetbulb_7d + palm_farmer_hh_ifls4:wetbulb_7d + ifls5:wetbulb_7d + palm_farmer_hh_ifls4:ifls5 + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode+pidlink+ifls5
* Individual FE additionally identify within-person changes; wave FE are absorbed.
reghdfe cesd_z c.wetbulb_7d c.palm_farmer_hh_ifls4#c.ifls5#c.wetbulb_7d c.palm_farmer_hh_ifls4#c.wetbulb_7d c.ifls5#c.wetbulb_7d c.palm_farmer_hh_ifls4#c.ifls5 $controls, absorb($fe stata_pidlink ifls5) $opts
eststo table_b_9

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 10; Wet-bulb / 3: \shortstack{Palm Shock\\Price Drop}
* Python: cesd_z ~ palm_price_gap_z * ifls5 * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z c.palm_price_gap_z##c.ifls5##c.wetbulb_7d $controls, absorb($fe) $opts
eststo table_b_10

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 11; Wet-bulb / 4: Urban Vehicle Owners
* Python: cesd_z ~ urban_vehicle_hh_ifls4 * post_subsidy * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
* Python df: wave5_df -> `if ifls5 == 1`
reghdfe cesd_z c.urban_vehicle_hh_ifls4##c.post_subsidy##c.wetbulb_7d $controls if ifls5 == 1, absorb($fe) $opts
eststo table_b_11

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 12; Wet-bulb / 5: \shortstack{Urban Vehicle Owners\\No Cash Transfer}
* Python: cesd_z ~ urban_vehicle_transfer_nonrecipient_ifls4 * post_subsidy * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
* Python df: wave5_df -> `if ifls5 == 1`
reghdfe cesd_z c.urban_vehicle_transfer__0dc7241a##c.post_subsidy##c.wetbulb_7d $controls if ifls5 == 1, absorb($fe) $opts
eststo table_b_12

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 13; Wet-bulb / 6: \shortstack{Urban Vehicle Owners\\Cash Transfer}
* Python: cesd_z ~ urban_vehicle_transfer_recipient_ifls4 * post_subsidy * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
* Python df: wave5_df -> `if ifls5 == 1`
reghdfe cesd_z c.urban_vehicle_transfer__60e2d6a2##c.post_subsidy##c.wetbulb_7d $controls if ifls5 == 1, absorb($fe) $opts
eststo table_b_13

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 14; Wet-bulb / 7: Job Loss
* Python: cesd_z ~ job_loss_180d * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z c.job_loss_180d##c.wetbulb_7d $controls, absorb($fe) $opts
eststo table_b_14

**** table_d: alternative palm-farmer definitions (IDs 1-3)
* code/analysis/tables_v2/table_d_palm_farmer_alt_definitions.py; table_d ID 1; Alternative definition / 1: palm_farmer_hh_ifls4
* Python: cesd_z ~ palm_farmer_hh_ifls4 * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z c.palm_farmer_hh_ifls4##c.ifls5##c.tmean_7d $controls, absorb($fe) $opts
eststo table_d_1

* code/analysis/tables_v2/table_d_palm_farmer_alt_definitions.py; table_d ID 2; Alternative definition / 2: palm_farmer_both_waves
* Python: cesd_z ~ palm_farmer_both_waves * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z c.palm_farmer_both_waves##c.ifls5##c.tmean_7d $controls, absorb($fe) $opts
eststo table_d_2

* code/analysis/tables_v2/table_d_palm_farmer_alt_definitions.py; table_d ID 3; Alternative definition / 3: palm_farmer_hh_ifls5
* Python: cesd_z ~ palm_farmer_hh_ifls5 * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
reghdfe cesd_z c.palm_farmer_hh_ifls5##c.ifls5##c.tmean_7d $controls, absorb($fe) $opts
eststo table_d_3

**** Export: one .tex (display) and one high-precision .csv (validation) per table.
* keep() lists only the displayed terms: heat, and for B/D the differential-heat interaction.
local out "output/tables/stata_port"
local tex `"se star b(%9.4f) se(%9.4f) stats(N N_clust, fmt(%9.0fc %9.0f)) nonumbers mtitles replace"'
local csv `"csv plain cells("b(fmt(%20.15g)) se(fmt(%20.15g))") stats(N N_clust, fmt(%20.15g)) nonumbers mtitles replace"'


local a_keep tmean_7d hot30_7d wetbulb_7d tmean_c tmax_c wetbulb_c
esttab table_a_1 table_a_2 table_a_3 table_a_4 table_a_5 table_a_6 using "`out'/table_a.tex", keep(`a_keep') `tex'
esttab table_a_1 table_a_2 table_a_3 table_a_4 table_a_5 table_a_6 using "`out'/table_a.csv", keep(`a_keep') `csv'
esttab table_a2_1 table_a2_2 table_a2_3 table_a2_4 table_a2_5 table_a2_6 using "`out'/table_a2.tex", keep(`a_keep') `tex'
esttab table_a2_1 table_a2_2 table_a2_3 table_a2_4 table_a2_5 table_a2_6 using "`out'/table_a2.csv", keep(`a_keep') `csv'

local b_temp_keep tmean_7d c.palm_farmer_hh_ifls4#c.ifls5#c.tmean_7d c.palm_price_gap_z#c.ifls5#c.tmean_7d c.urban_vehicle_hh_ifls4#c.post_subsidy#c.tmean_7d c.urban_vehicle_transfer__0dc7241a#c.post_subsidy#c.tmean_7d c.urban_vehicle_transfer__60e2d6a2#c.post_subsidy#c.tmean_7d c.job_loss_180d#c.tmean_7d
esttab table_b_1 table_b_2 table_b_3 table_b_4 table_b_5 table_b_6 table_b_7 using "`out'/table_b_temperature.tex", keep(`b_temp_keep') `tex'
esttab table_b_1 table_b_2 table_b_3 table_b_4 table_b_5 table_b_6 table_b_7 using "`out'/table_b_temperature.csv", keep(`b_temp_keep') `csv'

local b_wb_keep wetbulb_7d c.palm_farmer_hh_ifls4#c.ifls5#c.wetbulb_7d c.palm_price_gap_z#c.ifls5#c.wetbulb_7d c.urban_vehicle_hh_ifls4#c.post_subsidy#c.wetbulb_7d c.urban_vehicle_transfer__0dc7241a#c.post_subsidy#c.wetbulb_7d c.urban_vehicle_transfer__60e2d6a2#c.post_subsidy#c.wetbulb_7d c.job_loss_180d#c.wetbulb_7d
esttab table_b_8 table_b_9 table_b_10 table_b_11 table_b_12 table_b_13 table_b_14 using "`out'/table_b_wetbulb.tex", keep(`b_wb_keep') `tex'
esttab table_b_8 table_b_9 table_b_10 table_b_11 table_b_12 table_b_13 table_b_14 using "`out'/table_b_wetbulb.csv", keep(`b_wb_keep') `csv'

local d_keep tmean_7d c.palm_farmer_hh_ifls4#c.ifls5#c.tmean_7d c.palm_farmer_both_waves#c.ifls5#c.tmean_7d c.palm_farmer_hh_ifls5#c.ifls5#c.tmean_7d
esttab table_d_1 table_d_2 table_d_3 using "`out'/table_d.tex", keep(`d_keep') `tex'
esttab table_d_1 table_d_2 table_d_3 using "`out'/table_d.csv", keep(`d_keep') `csv'

log close
