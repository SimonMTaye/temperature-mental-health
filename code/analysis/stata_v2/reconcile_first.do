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

postclose results
log close
