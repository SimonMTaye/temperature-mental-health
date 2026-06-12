# Analysis Tables Summary

## `_lettered_common.py` — Shared Infrastructure
**Gist:** Shared helpers for all tables — loads data, fits models, formats LaTeX.

Loads the canonical analysis parquet (`30_analysis_table_input.parquet`), constructs kecamatan fixed effects and kabupaten cluster codes, defines the control set (age, female, edu_yrs, married, widowed), and provides helpers for fitting OLS models with `pyfixest` (clustered SEs at kabupaten), extracting coefficient/SE/p-value, computing linear combinations (`lincom`), formatting LaTeX cells, and writing `.tex`/`.csv` outputs. Key constants: `FE_POOLED = month + year + wave + kecamatan_fe`, `FE_IFLS5 = month + year + kecamatan_fe`, `CONTROLS = age + female + edu_yrs + married + widowed`.

---

## Table A — `table_a_headline.py`
**Gist:** Headline regressions — does heat amplify depression, and does the amplification get worse under each economic stressor?

Runs five OLS regressions of CES-D z-score on heat (7-day mean temperature deviation from local seasonal norm), each interacting heat with one of: job loss (12-month), palm-farmer x IFLS5 wave, coal-worker x IFLS5 wave, or fuel-subsidy-cut x urban-vehicle (IFLS5 only). All specs include kecamatan FE, month+year FE, wave FE (except the fuel cut, which is IFLS5-only), and demographic controls. Clusters at kabupaten. The five columns show the triple-interaction coefficient for each stressor, testing whether heat's effect on mental health is amplified for economically exposed households.

---

## Table B — `table_b_daynight.py`
**Gist:** Same headline interactions, but with max temperature (Tmax) and min temperature (Tmin) instead of mean temperature.

Two-panel robustness table. Panel A replaces the heat measure with `tmax_c_dev` (daily max temperature deviation) and Panel B with `tmin_c_dev` (daily min). Each panel runs the same four interaction regressions as Table A (job loss, palm, coal, fuel cut). Tests whether the amplification results are driven by daytime heat (Tmax) vs nighttime heat (Tmin), or both.

---

## Table C — `table_c_cdd.py`
**Gist:** Cooling-degree-day robustness — alternative heat thresholds (CDD above 30/32°C, Tmin above 23/24°C).

Four-panel robustness. Each panel uses a different cooling-degree-day or threshold-based heat measure (CDD Tmax>30, CDD Tmax>32, CDD Tmin>23, CDD Tmin>24) and runs the same four stressor interactions. Tests whether the headline results are robust to how heat is operationalized — continuous deviation vs. threshold-based accumulation of heat exposure.

---

## Table D — `table_d_sumstats.py`
**Gist:** Summary statistics for all key variables (means, SDs, N).

Computes descriptive statistics (mean, SD, p25, p50, p75, min, max, N) for five groups of variables: mental health outcomes (CES-D z-score, raw score, depression dummy), daily temperature exposure (mean, max, min, heat deviation), stressors and baseline groups (job loss, palm/coal/vehicle indicators), fuel-cut variables (post-subsidy, transport spending, IFLS5 only), and demographics. Reports on the panel sample (pooled waves) except fuel-cut variables which are IFLS5-only. Also reports the number of kabupaten clusters and kecamatan FE units.

---

## Table E — `table_e_cesd_decomp.py`
**Gist:** Decomposes the heat x stressor effect into CES-D sub-components (somatic, depressed affect, positive affect).

Runs the same four stressor interactions as Table A, but with three separate dependent variables: somatic/activity-related z-score, depressed affect z-score, and (reversed) positive affect z-score. Tests which dimension of depression (bodily symptoms, negative mood, or loss of positive affect) drives the amplification. Three panels, each with four columns.

---

## Table F — `table_f_mechanism_economic.py`
**Gist:** Economic mechanism — do the stressors affect labor income, non-labor income, and job loss?

Tests the first-stage economic channels. For palm shock, coal shock, and job loss separately, regresses: labor income (real wages), non-labor income, and job loss on the stressor. For the fuel cut, regresses transport spending and transport budget share (IHS-transformed). Confirms that the stressors have real economic bite (income drops, job loss) before attributing the mental health effect to economic distress. Ten rows in total across four panels.

---

## Table G — `table_g_mechanism_sleep.py`
**Gist:** Sleep mechanism — does heat (and heat x stressor) reduce sleep duration?

Regresses sleep duration (hours, IFLS5 only) on heat and its interactions with each stressor. Reports the heat main effect, the stressor main effect, and the heat x stressor interaction. Tests whether sleep disruption is a biological pathway through which heat amplifies depression under economic stress. Five columns: heat alone, then heat x job loss, palm, coal, and fuel cut.

---

## Table H — `table_h_within_day.py`
**Gist:** Within-day robustness — uses hourly temperature at the exact interview time instead of daily means.

Replaces the daily heat measure with the hourly temperature deviation at the survey hour (`heat_hr_dev`), and swaps month+year FE for calendar-day FE (since all interviewees on the same day share the same day, the day FE absorbs daily weather; identification now comes from within-day hourly variation). Runs the same five regressions as Table A. Tests whether the results survive this much tighter temporal control.

---

## Table I — `table_i_heat_window.py`
**Gist:** 7-day heat window robustness — alternative heat aggregations (7-day mean, days above 30°C, heatwave days).

Three-panel robustness using `tmean_7d_dev` (7-day mean temperature deviation), `hot30_7d_dev` (number of days above 30°C in past 7 days), and `heatwave_7d_dev` (local heatwave days). Each panel runs the four stressor interactions. Tests whether the results depend on how the trailing heat window is constructed.

---

## Table J — `table_j_job_loss_window.py`
**Gist:** Job-loss recall window sensitivity — how does the heat x job-loss interaction change as the recall window widens from 3 to 60 months?

Runs the heat x job_loss interaction eight times, each using a different recall window (3, 6, 9, 12, 18, 24, 36, 60 months). Reports the number treated, share of sample, the interaction coefficient, and the job-loss main effect. Tests whether the amplification is concentrated in recent job loss or persists for older job loss, and whether the 12-month headline window is arbitrary or robust.

---

## Table K — `table_k_job_loss_reason.py`
**Gist:** Job-loss reason heterogeneity — does the amplification differ for involuntary vs. family-related job loss?

Panel A runs separate regressions for any job loss, involuntary job loss, and family-related job loss. Panel B puts involuntary and family-related in the same model and computes a contrast (involuntary minus family) via linear combination. Reports treated count, % female, the interaction, and the main effect. Tests whether the heat amplification is specific to economically-driven job loss (involuntary) vs. life-event-driven (family).

---

## Table L — `table_l_temp_balance.py`
**Gist:** Balance check — are high-heat and low-heat interviewees similar on demographics, interview timing, and fuel-cut variables?

Splits the sample at the median 7-day heat deviation and compares high- vs. low-temperature groups on: interview month/day-of-year (Panel A), demographics (age, sex, education, marital status, household size, urban, PCE; Panel B), and fuel-cut variables (post-subsidy, cash transfer, BLT card; IFLS5 only, Panel C). Reports means, differences, normalized differences, and clustered p-values. Validates the research design by showing that who gets interviewed on hotter days is not systematically different.

---

## Table M — `table_m_palm_smallholder_mechanism.py`
**Gist:** Decomposes the palm amplification into self-employed smallholders (who bear CPO price risk) vs. other palm HHs (wage/family workers).

Reads raw IFLS employment data (`b3a_tk2`) to classify palm-farmer households into: pure self-employed smallholders (agriculture, self-employed, no wage work) and other palm HHs (wage workers, family workers, mixed). Runs separate and joint triple-interaction regressions (heat x IFLS5 x smallholder, heat x IFLS5 x other). Tests whether the amplification concentrates among smallholders who directly bear commodity price risk.

---

## Table N — `table_n_fuel_subsidy_card_mechanism.py`
**Gist:** Does the fuel-cut amplification concentrate in HHs without a cash-transfer buffer (BLT/PSKS card)?

Constructs `no_fuel_buffer` = 1 if the household received neither a cash transfer nor a BLT card. Runs five specs: (1) headline triple reminder, (2) quadruple interaction adding no-buffer, (3) triple within urban-vehicle HHs, (4) split-sample: urban-vehicle with no buffer, (5) split-sample: urban-vehicle with buffer. Tests whether the fuel-subsidy amplification is buffered by the government's compensatory cash transfer program.

---

## Table O — `table_o_sw_three_stressors.py`
**Gist:** Subjective wellbeing (non-depression) outcomes — do the stressors also affect life satisfaction, food/health adequacy, and happiness?

Reads raw IFLS subjective wellbeing items (`b3a_sw`) and regresses seven SW outcomes (Cantril ladder, living standard adequacy, food adequacy, health adequacy, family life, overall happiness, future standard-of-living expectation) on each stressor. Three columns: palm shock, fuel shock, and job loss. Tests whether the economic shocks have broader wellbeing effects beyond the clinical depression measure.

---

## Table P — `table_p_jobloss_slope_controls.py`
**Gist:** Robustness of heat x job-loss to slope controls — absorbs occupation-specific, health, age, and income heat sensitivity.

Sequentially adds heat-interacted controls to the job-loss regression: S1 (9 occupation-sector dummies), S2 (outdoor work dummy), S3 (baseline health proxies), S4 (centered age), S5 (log PCE as cooling-capacity proxy), S6 (all jointly). If the heat x job-loss coefficient survives S6, the amplification is not driven by construction workers or outdoor laborers being both more job-loss-prone and more heat-sensitive. Uses 7-day mean heat, IFLS4-baseline panel.

---

## Table Q — `table_q_palm_intensity_dose.py`
**Gist:** Dose-response — does the palm amplification scale with provincial palm acreage?

Replaces the binary palm-farmer flag with continuous provincial palm planted hectares (BPS data, wave-matched 2007/2014). Runs: binary reference, continuous z-scored intensity, and terciles (low/mid/high acreage). If the amplification scales with palm dependence — and is absent in low-palm provinces — it is harder to attribute to "something else changed in 2014." Uses `tmean_7d_dev` heat, panel sample.

---

## Table R — `table_r_palm_price_path.py`
**Gist:** Identifies the palm shock off the actual interview-date CPO price path instead of the binary wave dummy.

Replaces the IFLS5 wave dummy with the palm price each household faced at their interview date (USD/MT, z-scored). Tests five encodings: binary wave dummy (reference), within-IFLS5 price decline, pooled price level (no wave FE), pooled price with household FE, and IFLS4-to-IFLS5 household price gap. The honest finding (flagged in the docstring): pooled specs reproduce the effect, but the within-wave and household-gap specs are null — palm price is too collinear with the wave to separately identify.

---

## Table S — `table_s_palm_placebo.py`
**Gist:** Placebo tests — crop placebo (actual cultivation of rice/maize/etc.) and fake-shock permutation placebo.

Panel A runs the same wave-DiD triple for each crop household (rice, maize, cassava, groundnuts, soybean, coffee, rubber, identified by actual cultivation in `b2_ut1`) and reports the coefficient alongside each crop's 2007-2014 price change. Major staples should be null if the effect is crop-specific. Panel B randomly permutes palm-farmer status across households 200 times, re-estimates the triple each time, and computes a randomization-inference p-value. Tests whether the true palm coefficient sits outside the permutation null.

---

## Table T — `table_t_crop_palm_overlap.py`
**Gist:** Diagnostic — are the crop placebo effects from Table S independent of palm, or just palm overlap?

Because "palm farmer" = ag HH x palm province (no oil-palm crop code in IFLS), a maize grower in a palm province is also flagged as palm. Panel A shows overlap rates. Panel B splits crop triples by palm vs. non-palm province. Panel C horse-races crop and palm triples in the same model — if crop coefficients collapse while palm survives, the crop effects were palm overlap, not independent crop effects.

---

## Table U — `table_u_fuel_geography_robustness.py`
**Gist:** Fuel-cut geography robustness — FE ladder, pre/post balance, and placebo-in-time.

Panel A runs the headline fuel triple under a geography-FE ladder (province -> kabupaten -> kecamatan -> kabupaten x post -> kecamatan x post) to absorb any region-specific pre/post shift. Panel B checks pre- vs. post-subsidy balance on treatment, demographics, income, and heat within IFLS5. Panel C runs placebo-in-time: fake cutoff dates (Oct 1, 10, 20) inside the genuinely pre-hike window (before the real Nov 18 cut) — these should yield null triples, arguing against seasonal confounds.
