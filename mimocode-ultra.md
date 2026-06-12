# Tables in `code/analysis/tables/`

## table_a_headline.py — Headline heat × stressor interactions on CES-D
The main results table. Runs 5 regressions of CES-D z-score on heat deviation (`heat_c_dev`) interacted with each stressor: (1) heat alone, (2) heat × job loss, (3) heat × IFLS5 × palm farmer, (4) heat × IFLS5 × coal worker, (5) heat × post-subsidy × urban vehicle (IFLS5 only). Uses kecamatan FE, month+year FE, wave FE (pooled panel), kabupaten-clustered SE, and demographic controls. Outputs a 5-column LaTeX table with one coefficient per column.

## table_b_daynight.py — Day/night temperature robustness
Re-runs the 4 stressor interactions from Table A but replaces `heat_c_dev` with `tmax_c_dev` (Panel A) and `tmin_c_dev` (Panel B) to check whether daytime or nighttime temperature drives the result. Same FE structure, same stressors (job loss, palm, coal, fuel cut). Two panels × 4 columns.

## table_c_cdd.py — Cooling-degree-day robustness
Same 4 stressor interactions, but uses cooling-degree-day measures instead of raw temperature: CDD Tmax>30°C, Tmax>32°C, Tmin>23°C, Tmin>24°C. Each CDD measure gets its own sub-row within a single table, testing whether non-linear threshold-based heat exposure matters.

## table_d_sumstats.py — Summary statistics
Descriptive table for the analysis sample. Reports mean, SD for key variables grouped into 5 panels: (A) mental health outcomes (CES-D z, raw, depressed), (B) temperature measures, (C) stressors and baseline groups, (D) fuel-cut variables (IFLS5 only), (E) demographics. Also reports the number of kabupaten clusters and kecamatan FE units.

## table_e_cesd_decomp.py — CES-D subscale decomposition
Decomposes the headline CES-D result into 3 sub-scales: somatic/activity, depressed affect, and positive affect (all z-scored). For each subscale, runs the same 4 stressor interactions (job loss, palm, coal, fuel). Shows whether the heat×stressor effect loads on specific symptom clusters.

## table_f_mechanism_economic.py — Economic mechanism (income, job loss)
Tests the economic channel: does the palm/coal/fuel shock affect labor income, non-labor income, and job loss? 4 panels: (A) palm shock → labor/nonlabor/job loss, (B) coal shock → same, (C) job loss → labor/nonlabor, (D) fuel shock → transport spending and transport budget share. IFLS5-only for fuel panel, pooled for others.

## table_g_mechanism_sleep.py — Sleep mechanism
Tests sleep duration (hours) as a mediator. Runs the heat × stressor interactions on `sleep_dur_h` in IFLS5 only. Reports both the heat main effect and the heat × stressor interaction for each of the 5 stressors (heat alone, job loss, palm, coal, fuel cut).

## table_h_within-day.py — Within-day hourly temperature
Uses hourly temperature at the exact survey hour (`heat_hr_dev`) instead of daily means, with calendar-day FE + kecamatan FE (so it identifies off within-day temperature variation on the same day). Runs the 5 headline specs (heat, job loss, palm, coal, fuel cut) on CES-D z.

## table_i_heat_window.py — 7-day heat window robustness
Tests alternative heat window definitions: 7-day mean temperature deviation, days above 30°C, and local heatwave days. For each, runs the 4 stressor interactions. Checks whether the result is robust to how "heat exposure" is measured over a longer window.

## table_j_job_loss_window.py — Job-loss recall window sensitivity
Sweeps the job-loss recall window from 3 months to 60 months (8 windows). For each, reports the number treated, the share, and the heat × job-loss interaction coefficient. Shows how sensitive the heat×job-loss result is to the recall horizon. IFLS4-baseline panel only.

## table_k_job_loss_reason.py — Job-loss reason heterogeneity
Splits job loss into involuntary vs family-related reasons. Panel A: separate regressions for any/involuntary/family job loss. Panel B: joint regression with both subtypes plus a lincom contrast (involuntary − family) to test whether the heat amplification differs by reason. Reports % female among treated.

## table_l_temp_balance.py — Balance test on heat deviation
Validity check: splits the sample at the median 7-day heat deviation and tests whether high- vs low-temperature interviews differ on interview timing, demographics, and fuel-cut variables (cash transfer, BLT card). Reports group means, differences, normalized differences, and clustered p-values. If balanced, supports the identification assumption.

## table_m_palm_smallholder_mechanism.py — Palm smallholder vs other palm households
Decomposes the palm effect by household employment type using raw IFLS `b3a_tk2`. Creates `palm_self_pure_ifls4` (self-employed smallholder, bears CPO price risk) vs `palm_other_ifls4` (wage/family/mixed). Runs separate and joint triple-interaction regressions to see which subgroup drives the amplification.

## table_n_fuel_subsidy_card_mechanism.py — Fuel-cut: does cash-transfer buffer matter?
Tests whether the fuel-cut effect concentrates in urban-vehicle HHs that lacked a cash-transfer buffer (PSKS/BLT card). Runs 5 specs: headline triple, quadruple with no-buffer, within-urban-vehicle triple, and split-sample (no buffer vs with buffer). IFLS5 only.

## table_o_sw_three_stressors.py — Subjective wellbeing outcomes
Replaces CES-D with 7 subjective wellbeing items (Cantril ladder, living standard adequacy, food adequacy, health adequacy, family life, happiness, future expectations) from raw IFLS `b3a_sw`. For each SW outcome, runs the palm shock, fuel shock, and job loss regressions. Tests whether the stressors affect broader wellbeing, not just depressive symptoms.

## table_p_jobloss_slope_controls.py — Job-loss heat sensitivity with slope controls
Addresses confounding in the heat×job-loss interaction by adding heat-interacted controls sequentially: occupation sector dummies, outdoor work, baseline health, age, and log PCE (cooling capacity proxy). If the heat×job-loss coefficient survives these slope controls, the amplification is not just a proxy for occupation-specific or income-specific heat sensitivity.

## table_q_palm_intensity_dose.py — Palm intensity dose-response
Replaces the binary palm-farmer indicator with continuous provincial palm acreage (BPS data, wave-matched). Runs: binary reference, continuous z-score intensity, and terciles (low/mid/high acreage). Tests whether the heat amplification scales with how palm-dependent the province is.

## table_r_palm_price_path.py — Palm price path identification
Replaces the binary IFLS5 wave dummy with the actual palm price at interview date. Tests 5 price encodings: binary wave dummy, within-IFLS5 price decline, pooled price level (no wave FE), household FE (within-household price), and IFLS4→IFLS5 price gap. The honest finding: pooled specs work, but within-wave and household-level price variation alone are null because price is collinear with wave.

## table_s_palm_placebo.py — Palm placebo tests (crop placebo + permutation)
Panel A: runs the same wave-DiD triple for actual crop growers (rice, maize, cassava, groundnuts, soybean, coffee, rubber) and compares against their WB price changes. Major staples should be null if the effect is palm-specific. Panel B: randomly permutes palm-farmer status across HHs 200 times and computes a randomization-inference p-value for the true coefficient.

## table_t_crop_palm_overlap.py — Crop–palm overlap diagnostic
Diagnostic for Table S. Panel A: shows what % of each crop grower group lives in a palm province and is also flagged as a palm farmer. Panel B: splits the crop triple by palm vs non-palm province. Panel C: horse race — crop triple and palm triple in the same model. If crop coefficients collapse while palm survives, the crop effects are just palm overlap.

## table_u_fuel_geography_robustness.py — Fuel-cut geography robustness + placebo-in-time
Panel A: runs the fuel triple under a geography FE ladder (province → kabupaten → kecamatan → kab×post → kec×post) to absorb any region-specific pre/post shift. Panel B: pre vs post balance on treatment, demographics, and heat within IFLS5. Panel C: placebo-in-time — fake cutoff dates within the pre-hike window (Oct 2014) should yield null triples, arguing against seasonal confounds.
