version 19
clear all
set more off
* Run from repository root. All transformations and restrictions are in Python.
capture mkdir ".cache"
capture mkdir ".cache/stata_port"
log using ".cache/stata_port/port.log", text replace
adopath ++ ".cache/stata_port/ado"
which reghdfe
which ftools
postfile results str12 table_code int regression_id str80 coefficient str12 metric double value using ".cache/stata_port/stata_results.dta", replace
* PyFixest: month*year categorizes the numeric product, not calendar month-year.
* Numeric covariates (including ethnicity/religion) retain continuous coding.
* Singleton observations are recursively dropped in both engines.
* dof(clusters) mirrors PyFixest nonnested FE counts without pairwise rank adjustment.
* Identification assumes heat is conditionally unrelated to unobserved distress,
* given community FE, FE of numeric month*year, and included controls.

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a ID 1; CES-D / 1: tmean_7d
* Python: cesd_z ~ tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_a_1.dta", clear
reghdfe cesd_z tmean_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_a_1
post results ("table_a") (1) ("__model__") ("N") (e(N))
post results ("table_a") (1) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_a") (1) ("__model__") ("df_t") (e(df_r))
post results ("table_a") (1) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmean_7d
post results ("table_a") (1) ("heat") ("b") (r(estimate))
post results ("table_a") (1) ("heat") ("se") (r(se))
post results ("table_a") (1) ("heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_a_1_sample.csv", replace

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a ID 2; CES-D / 2: hot30_7d
* Python: cesd_z ~ hot30_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_a_2.dta", clear
reghdfe cesd_z hot30_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_a_2
post results ("table_a") (2) ("__model__") ("N") (e(N))
post results ("table_a") (2) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_a") (2) ("__model__") ("df_t") (e(df_r))
post results ("table_a") (2) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom hot30_7d
post results ("table_a") (2) ("heat") ("b") (r(estimate))
post results ("table_a") (2) ("heat") ("se") (r(se))
post results ("table_a") (2) ("heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_a_2_sample.csv", replace

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a ID 3; CES-D / 3: wetbulb_7d
* Python: cesd_z ~ wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_a_3.dta", clear
reghdfe cesd_z wetbulb_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_a_3
post results ("table_a") (3) ("__model__") ("N") (e(N))
post results ("table_a") (3) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_a") (3) ("__model__") ("df_t") (e(df_r))
post results ("table_a") (3) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom wetbulb_7d
post results ("table_a") (3) ("heat") ("b") (r(estimate))
post results ("table_a") (3) ("heat") ("se") (r(se))
post results ("table_a") (3) ("heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_a_3_sample.csv", replace

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a ID 4; CES-D / 4: tmean_c
* Python: cesd_z ~ tmean_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_a_4.dta", clear
reghdfe cesd_z tmean_c age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_a_4
post results ("table_a") (4) ("__model__") ("N") (e(N))
post results ("table_a") (4) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_a") (4) ("__model__") ("df_t") (e(df_r))
post results ("table_a") (4) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmean_c
post results ("table_a") (4) ("heat") ("b") (r(estimate))
post results ("table_a") (4) ("heat") ("se") (r(se))
post results ("table_a") (4) ("heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_a_4_sample.csv", replace

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a ID 5; CES-D / 5: tmax_c
* Python: cesd_z ~ tmax_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_a_5.dta", clear
reghdfe cesd_z tmax_c age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_a_5
post results ("table_a") (5) ("__model__") ("N") (e(N))
post results ("table_a") (5) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_a") (5) ("__model__") ("df_t") (e(df_r))
post results ("table_a") (5) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmax_c
post results ("table_a") (5) ("heat") ("b") (r(estimate))
post results ("table_a") (5) ("heat") ("se") (r(se))
post results ("table_a") (5) ("heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_a_5_sample.csv", replace

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a ID 6; CES-D / 6: wetbulb_c
* Python: cesd_z ~ wetbulb_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_a_6.dta", clear
reghdfe cesd_z wetbulb_c age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_a_6
post results ("table_a") (6) ("__model__") ("N") (e(N))
post results ("table_a") (6) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_a") (6) ("__model__") ("df_t") (e(df_r))
post results ("table_a") (6) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom wetbulb_c
post results ("table_a") (6) ("heat") ("b") (r(estimate))
post results ("table_a") (6) ("heat") ("se") (r(se))
post results ("table_a") (6) ("heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_a_6_sample.csv", replace

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a2 ID 1; Sleep / 1: tmean_7d
* Python: sleep_dur_h ~ tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_a2_1.dta", clear
reghdfe sleep_dur_h tmean_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_a2_1
post results ("table_a2") (1) ("__model__") ("N") (e(N))
post results ("table_a2") (1) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_a2") (1) ("__model__") ("df_t") (e(df_r))
post results ("table_a2") (1) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmean_7d
post results ("table_a2") (1) ("heat") ("b") (r(estimate))
post results ("table_a2") (1) ("heat") ("se") (r(se))
post results ("table_a2") (1) ("heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_a2_1_sample.csv", replace

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a2 ID 2; Sleep / 2: hot30_7d
* Python: sleep_dur_h ~ hot30_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_a2_2.dta", clear
reghdfe sleep_dur_h hot30_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_a2_2
post results ("table_a2") (2) ("__model__") ("N") (e(N))
post results ("table_a2") (2) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_a2") (2) ("__model__") ("df_t") (e(df_r))
post results ("table_a2") (2) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom hot30_7d
post results ("table_a2") (2) ("heat") ("b") (r(estimate))
post results ("table_a2") (2) ("heat") ("se") (r(se))
post results ("table_a2") (2) ("heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_a2_2_sample.csv", replace

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a2 ID 3; Sleep / 3: wetbulb_7d
* Python: sleep_dur_h ~ wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_a2_3.dta", clear
reghdfe sleep_dur_h wetbulb_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_a2_3
post results ("table_a2") (3) ("__model__") ("N") (e(N))
post results ("table_a2") (3) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_a2") (3) ("__model__") ("df_t") (e(df_r))
post results ("table_a2") (3) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom wetbulb_7d
post results ("table_a2") (3) ("heat") ("b") (r(estimate))
post results ("table_a2") (3) ("heat") ("se") (r(se))
post results ("table_a2") (3) ("heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_a2_3_sample.csv", replace

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a2 ID 4; Sleep / 4: tmean_c
* Python: sleep_dur_h ~ tmean_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_a2_4.dta", clear
reghdfe sleep_dur_h tmean_c age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_a2_4
post results ("table_a2") (4) ("__model__") ("N") (e(N))
post results ("table_a2") (4) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_a2") (4) ("__model__") ("df_t") (e(df_r))
post results ("table_a2") (4) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmean_c
post results ("table_a2") (4) ("heat") ("b") (r(estimate))
post results ("table_a2") (4) ("heat") ("se") (r(se))
post results ("table_a2") (4) ("heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_a2_4_sample.csv", replace

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a2 ID 5; Sleep / 5: tmax_c
* Python: sleep_dur_h ~ tmax_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_a2_5.dta", clear
reghdfe sleep_dur_h tmax_c age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_a2_5
post results ("table_a2") (5) ("__model__") ("N") (e(N))
post results ("table_a2") (5) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_a2") (5) ("__model__") ("df_t") (e(df_r))
post results ("table_a2") (5) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmax_c
post results ("table_a2") (5) ("heat") ("b") (r(estimate))
post results ("table_a2") (5) ("heat") ("se") (r(se))
post results ("table_a2") (5) ("heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_a2_5_sample.csv", replace

* code/analysis/tables_v2/table_a_temperature_effects.py; table_a2 ID 6; Sleep / 6: wetbulb_c
* Python: sleep_dur_h ~ wetbulb_c + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_a2_6.dta", clear
reghdfe sleep_dur_h wetbulb_c age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_a2_6
post results ("table_a2") (6) ("__model__") ("N") (e(N))
post results ("table_a2") (6) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_a2") (6) ("__model__") ("df_t") (e(df_r))
post results ("table_a2") (6) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom wetbulb_c
post results ("table_a2") (6) ("heat") ("b") (r(estimate))
post results ("table_a2") (6) ("heat") ("se") (r(se))
post results ("table_a2") (6) ("heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_a2_6_sample.csv", replace

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 1; Temperature / 1: Palm Shock
* Python: cesd_z ~ palm_farmer_hh_ifls4 * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_b_1.dta", clear
reghdfe cesd_z c.palm_farmer_hh_ifls4##c.ifls5##c.tmean_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_b_1
post results ("table_b") (1) ("__model__") ("N") (e(N))
post results ("table_b") (1) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_b") (1) ("__model__") ("df_t") (e(df_r))
post results ("table_b") (1) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmean_7d
post results ("table_b") (1) ("heat") ("b") (r(estimate))
post results ("table_b") (1) ("heat") ("se") (r(se))
post results ("table_b") (1) ("heat") ("p") (r(p))
lincom c.palm_farmer_hh_ifls4#c.ifls5#c.tmean_7d
post results ("table_b") (1) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_b") (1) ("differential_impact_heat") ("se") (r(se))
post results ("table_b") (1) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_b_1_sample.csv", replace

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 2; Temperature / 2: \shortstack{Palm Shocks\\Panel}
* Python: cesd_z ~ tmean_7d + palm_farmer_hh_ifls4:ifls5:tmean_7d + palm_farmer_hh_ifls4:tmean_7d + ifls5:tmean_7d + palm_farmer_hh_ifls4:ifls5 + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode+pidlink+ifls5
* Individual FE additionally identify within-person changes; wave FE are absorbed.
use "data/generated/stata_port/table_b_2.dta", clear
reghdfe cesd_z c.tmean_7d c.palm_farmer_hh_ifls4#c.ifls5#c.tmean_7d c.palm_farmer_hh_ifls4#c.tmean_7d c.ifls5#c.tmean_7d c.palm_farmer_hh_ifls4#c.ifls5 age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode stata_pidlink ifls5) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_b_2
post results ("table_b") (2) ("__model__") ("N") (e(N))
post results ("table_b") (2) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_b") (2) ("__model__") ("df_t") (e(df_r))
post results ("table_b") (2) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmean_7d
post results ("table_b") (2) ("heat") ("b") (r(estimate))
post results ("table_b") (2) ("heat") ("se") (r(se))
post results ("table_b") (2) ("heat") ("p") (r(p))
lincom c.palm_farmer_hh_ifls4#c.ifls5#c.tmean_7d
post results ("table_b") (2) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_b") (2) ("differential_impact_heat") ("se") (r(se))
post results ("table_b") (2) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_b_2_sample.csv", replace

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 3; Temperature / 3: \shortstack{Palm Shock\\Price Drop}
* Python: cesd_z ~ palm_price_gap_z * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_b_3.dta", clear
reghdfe cesd_z c.palm_price_gap_z##c.ifls5##c.tmean_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_b_3
post results ("table_b") (3) ("__model__") ("N") (e(N))
post results ("table_b") (3) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_b") (3) ("__model__") ("df_t") (e(df_r))
post results ("table_b") (3) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmean_7d
post results ("table_b") (3) ("heat") ("b") (r(estimate))
post results ("table_b") (3) ("heat") ("se") (r(se))
post results ("table_b") (3) ("heat") ("p") (r(p))
lincom c.palm_price_gap_z#c.ifls5#c.tmean_7d
post results ("table_b") (3) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_b") (3) ("differential_impact_heat") ("se") (r(se))
post results ("table_b") (3) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_b_3_sample.csv", replace

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 4; Temperature / 4: Urban Vehicle Owners
* Python: cesd_z ~ urban_vehicle_hh_ifls4 * post_subsidy * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_b_4.dta", clear
reghdfe cesd_z c.urban_vehicle_hh_ifls4##c.post_subsidy##c.tmean_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_b_4
post results ("table_b") (4) ("__model__") ("N") (e(N))
post results ("table_b") (4) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_b") (4) ("__model__") ("df_t") (e(df_r))
post results ("table_b") (4) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmean_7d
post results ("table_b") (4) ("heat") ("b") (r(estimate))
post results ("table_b") (4) ("heat") ("se") (r(se))
post results ("table_b") (4) ("heat") ("p") (r(p))
lincom c.urban_vehicle_hh_ifls4#c.post_subsidy#c.tmean_7d
post results ("table_b") (4) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_b") (4) ("differential_impact_heat") ("se") (r(se))
post results ("table_b") (4) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_b_4_sample.csv", replace

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 5; Temperature / 5: \shortstack{Urban Vehicle Owners\\No Cash Transfer}
* Python: cesd_z ~ urban_vehicle_transfer_nonrecipient_ifls4 * post_subsidy * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_b_5.dta", clear
reghdfe cesd_z c.urban_vehicle_transfer__0dc7241a##c.post_subsidy##c.tmean_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_b_5
post results ("table_b") (5) ("__model__") ("N") (e(N))
post results ("table_b") (5) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_b") (5) ("__model__") ("df_t") (e(df_r))
post results ("table_b") (5) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmean_7d
post results ("table_b") (5) ("heat") ("b") (r(estimate))
post results ("table_b") (5) ("heat") ("se") (r(se))
post results ("table_b") (5) ("heat") ("p") (r(p))
lincom c.urban_vehicle_transfer__0dc7241a#c.post_subsidy#c.tmean_7d
post results ("table_b") (5) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_b") (5) ("differential_impact_heat") ("se") (r(se))
post results ("table_b") (5) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_b_5_sample.csv", replace

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 6; Temperature / 6: \shortstack{Urban Vehicle Owners\\Cash Transfer}
* Python: cesd_z ~ urban_vehicle_transfer_recipient_ifls4 * post_subsidy * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_b_6.dta", clear
reghdfe cesd_z c.urban_vehicle_transfer__60e2d6a2##c.post_subsidy##c.tmean_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_b_6
post results ("table_b") (6) ("__model__") ("N") (e(N))
post results ("table_b") (6) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_b") (6) ("__model__") ("df_t") (e(df_r))
post results ("table_b") (6) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmean_7d
post results ("table_b") (6) ("heat") ("b") (r(estimate))
post results ("table_b") (6) ("heat") ("se") (r(se))
post results ("table_b") (6) ("heat") ("p") (r(p))
lincom c.urban_vehicle_transfer__60e2d6a2#c.post_subsidy#c.tmean_7d
post results ("table_b") (6) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_b") (6) ("differential_impact_heat") ("se") (r(se))
post results ("table_b") (6) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_b_6_sample.csv", replace

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 7; Temperature / 7: Job Loss
* Python: cesd_z ~ job_loss_180d * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_b_7.dta", clear
reghdfe cesd_z c.job_loss_180d##c.tmean_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_b_7
post results ("table_b") (7) ("__model__") ("N") (e(N))
post results ("table_b") (7) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_b") (7) ("__model__") ("df_t") (e(df_r))
post results ("table_b") (7) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmean_7d
post results ("table_b") (7) ("heat") ("b") (r(estimate))
post results ("table_b") (7) ("heat") ("se") (r(se))
post results ("table_b") (7) ("heat") ("p") (r(p))
lincom c.job_loss_180d#c.tmean_7d
post results ("table_b") (7) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_b") (7) ("differential_impact_heat") ("se") (r(se))
post results ("table_b") (7) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_b_7_sample.csv", replace

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 8; Wet-bulb / 1: Palm Shock
* Python: cesd_z ~ palm_farmer_hh_ifls4 * ifls5 * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_b_8.dta", clear
reghdfe cesd_z c.palm_farmer_hh_ifls4##c.ifls5##c.wetbulb_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_b_8
post results ("table_b") (8) ("__model__") ("N") (e(N))
post results ("table_b") (8) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_b") (8) ("__model__") ("df_t") (e(df_r))
post results ("table_b") (8) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom wetbulb_7d
post results ("table_b") (8) ("heat") ("b") (r(estimate))
post results ("table_b") (8) ("heat") ("se") (r(se))
post results ("table_b") (8) ("heat") ("p") (r(p))
lincom c.palm_farmer_hh_ifls4#c.ifls5#c.wetbulb_7d
post results ("table_b") (8) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_b") (8) ("differential_impact_heat") ("se") (r(se))
post results ("table_b") (8) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_b_8_sample.csv", replace

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 9; Wet-bulb / 2: \shortstack{Palm Shocks\\Panel}
* Python: cesd_z ~ wetbulb_7d + palm_farmer_hh_ifls4:ifls5:wetbulb_7d + palm_farmer_hh_ifls4:wetbulb_7d + ifls5:wetbulb_7d + palm_farmer_hh_ifls4:ifls5 + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode+pidlink+ifls5
* Individual FE additionally identify within-person changes; wave FE are absorbed.
use "data/generated/stata_port/table_b_9.dta", clear
reghdfe cesd_z c.wetbulb_7d c.palm_farmer_hh_ifls4#c.ifls5#c.wetbulb_7d c.palm_farmer_hh_ifls4#c.wetbulb_7d c.ifls5#c.wetbulb_7d c.palm_farmer_hh_ifls4#c.ifls5 age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode stata_pidlink ifls5) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_b_9
post results ("table_b") (9) ("__model__") ("N") (e(N))
post results ("table_b") (9) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_b") (9) ("__model__") ("df_t") (e(df_r))
post results ("table_b") (9) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom wetbulb_7d
post results ("table_b") (9) ("heat") ("b") (r(estimate))
post results ("table_b") (9) ("heat") ("se") (r(se))
post results ("table_b") (9) ("heat") ("p") (r(p))
lincom c.palm_farmer_hh_ifls4#c.ifls5#c.wetbulb_7d
post results ("table_b") (9) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_b") (9) ("differential_impact_heat") ("se") (r(se))
post results ("table_b") (9) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_b_9_sample.csv", replace

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 10; Wet-bulb / 3: \shortstack{Palm Shock\\Price Drop}
* Python: cesd_z ~ palm_price_gap_z * ifls5 * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_b_10.dta", clear
reghdfe cesd_z c.palm_price_gap_z##c.ifls5##c.wetbulb_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_b_10
post results ("table_b") (10) ("__model__") ("N") (e(N))
post results ("table_b") (10) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_b") (10) ("__model__") ("df_t") (e(df_r))
post results ("table_b") (10) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom wetbulb_7d
post results ("table_b") (10) ("heat") ("b") (r(estimate))
post results ("table_b") (10) ("heat") ("se") (r(se))
post results ("table_b") (10) ("heat") ("p") (r(p))
lincom c.palm_price_gap_z#c.ifls5#c.wetbulb_7d
post results ("table_b") (10) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_b") (10) ("differential_impact_heat") ("se") (r(se))
post results ("table_b") (10) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_b_10_sample.csv", replace

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 11; Wet-bulb / 4: Urban Vehicle Owners
* Python: cesd_z ~ urban_vehicle_hh_ifls4 * post_subsidy * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_b_11.dta", clear
reghdfe cesd_z c.urban_vehicle_hh_ifls4##c.post_subsidy##c.wetbulb_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_b_11
post results ("table_b") (11) ("__model__") ("N") (e(N))
post results ("table_b") (11) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_b") (11) ("__model__") ("df_t") (e(df_r))
post results ("table_b") (11) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom wetbulb_7d
post results ("table_b") (11) ("heat") ("b") (r(estimate))
post results ("table_b") (11) ("heat") ("se") (r(se))
post results ("table_b") (11) ("heat") ("p") (r(p))
lincom c.urban_vehicle_hh_ifls4#c.post_subsidy#c.wetbulb_7d
post results ("table_b") (11) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_b") (11) ("differential_impact_heat") ("se") (r(se))
post results ("table_b") (11) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_b_11_sample.csv", replace

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 12; Wet-bulb / 5: \shortstack{Urban Vehicle Owners\\No Cash Transfer}
* Python: cesd_z ~ urban_vehicle_transfer_nonrecipient_ifls4 * post_subsidy * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_b_12.dta", clear
reghdfe cesd_z c.urban_vehicle_transfer__0dc7241a##c.post_subsidy##c.wetbulb_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_b_12
post results ("table_b") (12) ("__model__") ("N") (e(N))
post results ("table_b") (12) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_b") (12) ("__model__") ("df_t") (e(df_r))
post results ("table_b") (12) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom wetbulb_7d
post results ("table_b") (12) ("heat") ("b") (r(estimate))
post results ("table_b") (12) ("heat") ("se") (r(se))
post results ("table_b") (12) ("heat") ("p") (r(p))
lincom c.urban_vehicle_transfer__0dc7241a#c.post_subsidy#c.wetbulb_7d
post results ("table_b") (12) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_b") (12) ("differential_impact_heat") ("se") (r(se))
post results ("table_b") (12) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_b_12_sample.csv", replace

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 13; Wet-bulb / 6: \shortstack{Urban Vehicle Owners\\Cash Transfer}
* Python: cesd_z ~ urban_vehicle_transfer_recipient_ifls4 * post_subsidy * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_b_13.dta", clear
reghdfe cesd_z c.urban_vehicle_transfer__60e2d6a2##c.post_subsidy##c.wetbulb_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_b_13
post results ("table_b") (13) ("__model__") ("N") (e(N))
post results ("table_b") (13) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_b") (13) ("__model__") ("df_t") (e(df_r))
post results ("table_b") (13) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom wetbulb_7d
post results ("table_b") (13) ("heat") ("b") (r(estimate))
post results ("table_b") (13) ("heat") ("se") (r(se))
post results ("table_b") (13) ("heat") ("p") (r(p))
lincom c.urban_vehicle_transfer__60e2d6a2#c.post_subsidy#c.wetbulb_7d
post results ("table_b") (13) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_b") (13) ("differential_impact_heat") ("se") (r(se))
post results ("table_b") (13) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_b_13_sample.csv", replace

* code/analysis/tables_v2/table_b_temperature_and_shock_effects.py; table_b ID 14; Wet-bulb / 7: Job Loss
* Python: cesd_z ~ job_loss_180d * wetbulb_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_b_14.dta", clear
reghdfe cesd_z c.job_loss_180d##c.wetbulb_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_b_14
post results ("table_b") (14) ("__model__") ("N") (e(N))
post results ("table_b") (14) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_b") (14) ("__model__") ("df_t") (e(df_r))
post results ("table_b") (14) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom wetbulb_7d
post results ("table_b") (14) ("heat") ("b") (r(estimate))
post results ("table_b") (14) ("heat") ("se") (r(se))
post results ("table_b") (14) ("heat") ("p") (r(p))
lincom c.job_loss_180d#c.wetbulb_7d
post results ("table_b") (14) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_b") (14) ("differential_impact_heat") ("se") (r(se))
post results ("table_b") (14) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_b_14_sample.csv", replace

* code/analysis/tables_v2/table_d_palm_farmer_alt_definitions.py; table_d ID 1; Alternative definition / 1: palm_farmer_hh_ifls4
* Python: cesd_z ~ palm_farmer_hh_ifls4 * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_d_1.dta", clear
reghdfe cesd_z c.palm_farmer_hh_ifls4##c.ifls5##c.tmean_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_d_1
post results ("table_d") (1) ("__model__") ("N") (e(N))
post results ("table_d") (1) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_d") (1) ("__model__") ("df_t") (e(df_r))
post results ("table_d") (1) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmean_7d
post results ("table_d") (1) ("heat") ("b") (r(estimate))
post results ("table_d") (1) ("heat") ("se") (r(se))
post results ("table_d") (1) ("heat") ("p") (r(p))
lincom c.palm_farmer_hh_ifls4#c.ifls5#c.tmean_7d
post results ("table_d") (1) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_d") (1) ("differential_impact_heat") ("se") (r(se))
post results ("table_d") (1) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_d_1_sample.csv", replace

* code/analysis/tables_v2/table_d_palm_farmer_alt_definitions.py; table_d ID 2; Alternative definition / 2: palm_farmer_both_waves
* Python: cesd_z ~ palm_farmer_both_waves * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_d_2.dta", clear
reghdfe cesd_z c.palm_farmer_both_waves##c.ifls5##c.tmean_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_d_2
post results ("table_d") (2) ("__model__") ("N") (e(N))
post results ("table_d") (2) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_d") (2) ("__model__") ("df_t") (e(df_r))
post results ("table_d") (2) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmean_7d
post results ("table_d") (2) ("heat") ("b") (r(estimate))
post results ("table_d") (2) ("heat") ("se") (r(se))
post results ("table_d") (2) ("heat") ("p") (r(p))
lincom c.palm_farmer_both_waves#c.ifls5#c.tmean_7d
post results ("table_d") (2) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_d") (2) ("differential_impact_heat") ("se") (r(se))
post results ("table_d") (2) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_d_2_sample.csv", replace

* code/analysis/tables_v2/table_d_palm_farmer_alt_definitions.py; table_d ID 3; Alternative definition / 3: palm_farmer_hh_ifls5
* Python: cesd_z ~ palm_farmer_hh_ifls5 * ifls5 * tmean_7d + age + female + edu_yrs_nullable + married + widowed + divorced + ethnicity_nullable + religion_nullable | month*year+gadm_fullcode
use "data/generated/stata_port/table_d_3.dta", clear
reghdfe cesd_z c.palm_farmer_hh_ifls5##c.ifls5##c.tmean_7d age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable, absorb(stata_month_times_year stata_gadm_fullcode) vce(cluster kabupaten_full_code) dof(clusters) tolerance(1e-12)
estimates store table_d_3
post results ("table_d") (3) ("__model__") ("N") (e(N))
post results ("table_d") (3) ("__model__") ("n_clusters") (e(N_clust))
post results ("table_d") (3) ("__model__") ("df_t") (e(df_r))
post results ("table_d") (3) ("__model__") ("df_k") (e(df_m)+e(df_a)+1)
lincom tmean_7d
post results ("table_d") (3) ("heat") ("b") (r(estimate))
post results ("table_d") (3) ("heat") ("se") (r(se))
post results ("table_d") (3) ("heat") ("p") (r(p))
lincom c.palm_farmer_hh_ifls5#c.ifls5#c.tmean_7d
post results ("table_d") (3) ("differential_impact_heat") ("b") (r(estimate))
post results ("table_d") (3) ("differential_impact_heat") ("se") (r(se))
post results ("table_d") (3) ("differential_impact_heat") ("p") (r(p))
export delimited stata_source_row pidlink wave if e(sample) using ".cache/stata_port/table_d_3_sample.csv", replace

postclose results
use ".cache/stata_port/stata_results.dta", clear
export delimited using ".cache/stata_port/stata_results.csv", replace
log close
