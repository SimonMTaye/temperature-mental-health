# Code/Analysis Tables Summary

All scripts in `code/analysis/tables` are standalone table builders. Unless noted, they read `data/generated/30_analysis_table_input.parquet`, build kecamatan FE and kabupaten cluster codes via `_lettered_common.py`, fit `pyfixest.feols`, then write `.tex`, `_body.tex`, and `.csv` to `output/tables/`.

## Table A, `table_a_headline.py`

Gist: Main CES-D heat result plus the four headline stressor amplifications.

Detail: Recreate it by restricting to respondents with IFLS4 baseline stressor indicators, then estimating CES-D z-score on `heat_c_dev` and heat interactions with job loss, IFLS5 x palm farmer, IFLS5 x coal worker, and IFLS5-only post-subsidy x urban vehicle household. The pooled specs use demographic controls, kecamatan FE, month/year/wave FE, and kabupaten-clustered SE; the fuel column uses IFLS5 only with month/year/kecamatan FE.

## Table B, `table_b_daynight.py`

Gist: Replaces mean heat with day and night temperature deviations.

Detail: It reruns the four stressor interaction columns from Table A using `tmax_c_dev` and `tmin_c_dev`. Each panel reports the heat-measure x stressor coefficient for job loss, palm, coal, and fuel cut, keeping the same controls, FE structure, sample logic, and clustering as Table A.

## Table C, `table_c_cdd.py`

Gist: Checks whether results survive threshold-based heat exposure measures.

Detail: It reruns the Table B interaction grid using cooling-degree-day style variables: `cdd_tmax30`, `cdd_tmax32`, `cdd_tmin23`, and `cdd_tmin24`. For each threshold measure, it estimates CES-D on heat-threshold x stressor interactions with the same pooled or IFLS5-only fixed effects.

## Table D, `table_d_sumstats.py`

Gist: Summary statistics for outcomes, heat, stressors, fuel variables, and demographics.

Detail: It loads the canonical panel, applies the standard model-frame restriction, and computes mean, SD, quantiles, min, max, and N for CES-D outcomes, daily temperature variables, stressor indicators, IFLS5 fuel variables, and controls. Fuel-cut variables are summarized only in IFLS5; the rest use the pooled panel.

## Table E, `table_e_cesd_decomp.py`

Gist: Decomposes the CES-D result into symptom subcomponents.

Detail: It repeats the four stressor interaction models using `somatic_z`, `depraffect_z`, and `posaffect_z` as dependent variables. The point is to see whether heat-amplified distress loads on somatic/activity symptoms, depressed affect, or positive affect rather than only the aggregate CES-D index.

## Table F, `table_f_mechanism_economic.py`

Gist: Tests whether the stressors show up in economic outcomes.

Detail: It builds `palm_shock`, `coal_shock`, and `fuel_shock`, then regresses labor income, nonlabor income, job loss, transport spending, or transport share on the relevant shock indicators. This is not a heat-interaction table; it is a mechanism check that asks whether palm, coal, job-loss, and fuel-cut shocks move economic channels.

## Table G, `table_g_mechanism_sleep.py`

Gist: Tests sleep duration as a heat-stress mechanism.

Detail: It restricts to IFLS5 and estimates `sleep_dur_h` on `heat_c_dev`, then adds interactions with job loss, palm baseline, coal baseline, and fuel-cut exposure. It reports both the heat x stressor coefficient and the direct heat coefficient under IFLS5 month/year/kecamatan FE.

## Table H, `table_h_within_day.py`

Gist: Hourly, within-calendar-day heat robustness for CES-D.

Detail: It uses `heat_hr_dev` and `tmean_c_hour` instead of daily heat, with calendar-day FE plus kecamatan FE. It reruns the Table A style heat/stressor interactions, but explicitly errors if the canonical hourly columns are entirely missing, so recreating it requires those hourly fields in the parquet.

## Table I, `table_i_heat_window.py`

Gist: Seven-day heat-window robustness.

Detail: It reruns the four stressor interactions using `tmean_7d_dev`, `hot30_7d_dev`, and `heatwave_7d_dev`. This asks whether the Table A pattern is about same-day heat only or broader recent heat exposure.

## Table J, `table_j_job_loss_window.py`

Gist: Varies the recall window for job loss.

Detail: It keeps respondents who appear in IFLS4, then estimates `cesd_z ~ heat_c_dev * job_loss_window` for windows from 3 months through 60 months. It reports treated counts, treated shares, the heat x job-loss coefficient, and the job-loss main effect.

## Table K, `table_k_job_loss_reason.py`

Gist: Splits job loss by reason.

Detail: It compares any job loss, involuntary job loss, and family-related job loss. It first runs separate heat x reason regressions, then a joint model with involuntary and family-related losses together and a linear contrast of the two heat interactions.

## Table L, `table_l_temp_balance.py`

Gist: Balance table for high versus low seven-day heat exposure.

Detail: It splits pooled and IFLS5 samples at the median `tmean_7d_dev`, then compares interview timing, demographics, and fuel-cut compensation variables across low/high temperature interviews. It reports means, high-minus-low differences, clustered p-values, normalized differences, and row-level Ns.

## Table M, `table_m_palm_smallholder_mechanism.py`

Gist: Splits the palm effect into smallholder versus other palm households.

Detail: It reads raw IFLS `b3a_tk2.dta`, derives household agricultural self-employment and wage indicators from sector/status fields, carries IFLS4 status forward, and splits `palm_farmer_hh_ifls4` into pure self-employed smallholders versus other palm households. It estimates separate and joint heat x IFLS5 x palm-subtype triples.

## Table N, `table_n_fuel_subsidy_card_mechanism.py`

Gist: Tests whether cash-transfer buffering matters for the fuel-cut effect.

Detail: It builds `has_fuel_buffer` from `cash_transfer_recipient` or `blt_card`, then runs the headline IFLS5 fuel triple, a quadruple with `no_fuel_buffer`, a within-urban-vehicle triple, and split-sample heat x post-subsidy models for buffered versus unbuffered urban-vehicle households.

## Table O, `table_o_sw_three_stressors.py`

Gist: Uses subjective wellbeing items as alternative outcomes.

Detail: It reads raw IFLS `b3a_sw.dta`, cleans invalid response codes, merges items onto the canonical panel, and estimates direct stressor effects on wellbeing: palm x IFLS5, fuel shock, and job loss. This is a channel/validation table, not a heat-interaction table.

## Table P, `table_p_jobloss_slope_controls.py`

Gist: Checks whether heat x job loss survives heat-slope confound controls.

Detail: It reads IFLS4 sector from `b3a_tk2.dta`, builds occupation-sector dummies, outdoor-work, health, age, and log-PCE slope controls, then adds them interacted with `tmean_7d_dev` one block at a time. The target coefficient is always `tmean_7d_dev:job_loss_1_yr`.

## Table Q, `table_q_palm_intensity_dose.py`

Gist: Replaces binary palm exposure with continuous provincial palm acreage.

Detail: It merges `data/raw/palm_area_prov_BPS.csv`, creates wave-matched palm acreage, log-acreage z-score intensity, and palm-acreage terciles. It estimates heat x IFLS5 x palm intensity models to see whether the palm effect scales with local palm dependence.

## Table R, `table_r_palm_price_path.py`

Gist: Tests whether palm results identify off the palm price path rather than only IFLS5.

Detail: It uses canonical `palm_price_usd_mt` and `palm_price_z` to build price-decline, negative-price, and household price-gap measures. It compares the binary IFLS5 palm triple with within-IFLS5 price timing, pooled price-level, household-FE price, and IFLS4-to-IFLS5 price-gap specifications.

## Table S, `table_s_palm_placebo.py`

Gist: Palm placebo tests using other crops and fake palm assignment.

Detail: It reads raw IFLS crop cultivation from `b2_ut1.dta`, builds IFLS4 baseline grower indicators for rice, maize, cassava, groundnuts, soybean, coffee, and rubber, and runs the same heat x IFLS5 x grower triple. It also permutes palm-farmer status across households 200 times to build a randomization placebo distribution.

## Table T, `table_t_crop_palm_overlap.py`

Gist: Diagnoses whether crop placebo effects are really palm-overlap effects.

Detail: It reads crop cultivation from `b2_ut1.dta`, computes crop grower overlap with palm provinces and the palm flag, then estimates crop triples in all, palm, and non-palm provinces. It also horse-races each crop triple against the palm triple in the same model.

## Table U, `table_u_fuel_geography_robustness.py`

Gist: Fuel-cut geography and timing robustness.

Detail: It restricts to IFLS5 and tests the heat x urban-vehicle x post-subsidy triple under increasingly demanding geography FE, including kabupaten x post and kecamatan x post. It also reports pre/post balance and placebo cutoffs before the real November 18, 2014 fuel hike.
