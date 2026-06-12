# Analysis Tables Documentation

Each `table_<letter>_*.py` is a standalone script that writes LaTeX (`*.tex`, `*_body.tex`) and a `*.csv` to `output/tables/`. Shared helpers live in `_lettered_common.py`.

---

### `_lettered_common.py`
**Gist:** Shared helper library — loads data, fits fixed-effect OLS models via `pyfixest`, formats LaTeX cells, and writes outputs.

**Detail:** Provides the canonical data loader (`load_analysis()` reads `data/generated/30_analysis_table_input.parquet`), creates kecamatan and kabupaten administrative codes, restricts to the analysis panel (rows with IFLS4 baseline group indicators), drops singleton fixed-effect groups, fits `pyfixest.feols` models with kabupaten-clustered standard errors (CRV1), and provides helpers for extracting coefficient/SE/p-value from fitted models, computing linear combinations (`lincom`), formatting significance stars, and writing `.tex` + `.csv` outputs to `output/tables/`. All tables import from this.

---

### Table A — `table_a_headline.py`
**Gist:** The headline result — heat amplifies the mental-health effect of four economic stressors (job loss, palm shock, coal shock, fuel subsidy cut).

**Detail:** Runs five regressions of CES-D z-score on heat deviation (`heat_c_dev`) interacted with each stressor. Columns (1)-(4) use the pooled IFLS4+IFLS5 panel with kecamatan/month/year/wave FE and demographic controls: (1) heat × job loss (12-month recall), (2) heat × IFLS5 × palm farmer HH, (3) heat × IFLS5 × coal worker HH, (4) heat × post-subsidy-cut × urban vehicle HH (IFLS5 only, no wave FE). Each triple interaction captures whether heat's mental-health toll is amplified for households hit by the respective shock. Outputs a LaTeX table with coefficients, clustered SEs, and observation counts.

---

### Table B — `table_b_daynight.py`
**Gist:** Robustness check — does the headline result hold using daytime max temp or nighttime min temp instead of mean temp?

**Detail:** Replicates Table A's four stressor interactions but replaces `heat_c_dev` with `tmax_c_dev` (Panel A: max temperature deviation) and `tmin_c_dev` (Panel B: min temperature deviation). Each panel runs the same four regressions (job loss, palm, coal, fuel cut) to check whether the amplification effect is driven by daytime heat, nighttime heat, or both. Uses the same FE structure as Table A.

---

### Table C — `table_c_cdd.py`
**Gist:** Robustness check — does the headline result hold using cooling-degree-day (CDD) measures instead of simple temperature deviation?

**Detail:** Uses four alternative heat measures: CDD with Tmax > 30°C, Tmax > 32°C, Tmin > 23°C, and Tmin > 24°C. For each, runs the same four stressor interactions from Table A. This tests whether the result is sensitive to how "extreme heat" is defined — continuous deviation vs. threshold-based cumulative exposure. Imports `run_spec` from Table B to reuse the specification runner.

---

### Table D — `table_d_sumstats.py`
**Gist:** Summary statistics — means, SDs, quartiles, and N for all key variables in the analysis sample.

**Detail:** Produces a descriptive table with five panels: (A) mental health outcomes (CES-D z-score, raw score, depressed indicator), (B) daily temperature exposure (mean, heat deviation, max, min), (C) stressors and baseline groups (job loss, palm farmer, coal worker, urban vehicle HH), (D) fuel-cut variables (post subsidy, transport share/spending — IFLS5 only), (E) demographic controls (age, female, education, married, widowed). Each variable reports mean, SD, p25, p50, p75, min, max, and N. Also reports the number of kabupaten clusters and kecamatan FE units.

---

### Table E — `table_e_cesd_decomp.py`
**Gist:** Decomposes the headline result into three CES-D subscales — somatic, depressed affect, and positive affect.

**Detail:** Runs the same four stressor interactions from Table A but with three different dependent variables: somatic/activity-related z-score, depressed affect z-score, and positive affect z-score. This shows which dimension of depression is most affected by the heat × stressor interaction. Same FE and controls as Table A; columns are job loss, palm shock, coal shock, fuel cut.

---

### Table F — `table_f_mechanism_economic.py`
**Gist:** Economic mechanism — do the shock treatments actually move income, non-labor income, job loss, and transport spending?

**Detail:** First-stage/mechanism regressions in four panels: (A) palm shock's effect on labor income, non-labor income, and job loss; (B) coal shock's effect on the same three outcomes; (C) job loss's effect on labor and non-labor income; (D) fuel cut's effect on transport spending and transport budget share (IHS-transformed). Each regression is a simple shock → outcome model with kecamatan/month/year/wave FE. This establishes that the shocks have real economic bite, not just statistical interaction with heat.

---

### Table G — `table_g_mechanism_sleep.py`
**Gist:** Sleep mechanism — does heat reduce sleep, and does that reduction amplify under each stressor?

**Detail:** Uses sleep duration (hours) as the dependent variable instead of CES-D. Column (1) tests the main effect of heat on sleep; columns (2)-(5) add each stressor interaction (job loss, palm, coal, fuel cut). Reports both the interaction term and the main heat coefficient. This tests whether disrupted sleep is a channel through which heat × stressor harms mental health. IFLS5 sample only (sleep data only in IFLS5), kecamatan/month/year FE.

---

### Table H — `table_h_within_day.py`
**Gist:** Within-day robustness — does the result hold using hourly temperature at the exact survey interview time, with calendar-day FE?

**Detail:** Replaces the daily heat measure with `heat_hr_dev` (hourly temperature deviation at the survey hour) and swaps month+year FE for calendar-day FE (`day_id`). This absorbs all day-level variation and identifies only off within-day temperature differences across interviews happening at different hours on the same day in the same kecamatan. Runs the same five specs as Table A (main effect + four stressor interactions).

---

### Table I — `table_i_heat_window.py`
**Gist:** Robustness check — does the headline result hold using 7-day heat windows instead of single-day temperature?

**Detail:** Uses three alternative heat measures: 7-day mean temperature deviation (`tmean_7d_dev`), number of days above 30°C in the past 7 days (`hot30_7d_dev`), and local heatwave days in the past 7 days (`heatwave_7d_dev`). For each, runs the four stressor interactions. This tests whether cumulative multi-day heat exposure matters, not just the single interview-day temperature.

---

### Table J — `table_j_job_loss_window.py`
**Gist:** Recall-window sensitivity — how does the heat × job-loss interaction change as you widen the job-loss recall window from 3 months to 5 years?

**Detail:** Runs the heat × job loss interaction eight times, varying the recall window: 3, 6, 9, 12, 18, 24, 36, and 60 months. Reports the interaction coefficient, the treated count, the share of the sample with job loss, and the main effect of job loss at each window. Uses IFLS4-baseline panel with kecamatan/month/year/wave FE. This shows how sensitive the result is to how "recent" job loss is defined.

---

### Table K — `table_k_job_loss_reason.py`
**Gist:** Job-loss heterogeneity — does the heat amplification differ by job-loss reason (involuntary vs. family-related)?

**Detail:** Panel A runs separate regressions for any job loss, involuntary job loss, and family-related job loss, each interacted with heat. Reports treated N, % female, interaction coefficient, and stressor main effect. Panel B puts involuntary and family-related loss in the same regression and computes a linear contrast (involuntary minus family) to test whether the two subtypes differ statistically. IFLS4-baseline panel with kecamatan/month/year/wave FE.

---

### Table L — `table_l_temp_balance.py`
**Gist:** Balance test — are high-heat and low-heat interviewees similar on demographics, timing, and fuel-cut variables?

**Detail:** Splits the sample at the median 7-day heat deviation and compares high vs. low groups on: (A) interview timing (month, day-of-year, wave), (B) demographics (age, sex, education, marital status, HH size, urban, log PCE, urban-vehicle baseline), (C) fuel-cut variables (post-subsidy, cash transfer recipient, BLT card — IFLS5 only). Reports group means, high-minus-low difference, normalized difference, and kabupaten-clustered p-value. A valid design requires these to be balanced — the identifying variation comes from within-kecamatan-month-year weather wiggles.

---

### Table M — `table_m_palm_smallholder_mechanism.py`
**Gist:** Decomposes the palm effect into self-employed smallholders (who bear CPO price risk) vs. other palm HHs (wage/family workers).

**Detail:** Reads raw IFLS `b3a_tk2.dta` to identify household agricultural employment status (self-employed vs. wage worker in agriculture, sector code 1). Constructs `palm_self_pure_ifls4` (palm farmer HH + self-employed + not wage worker) and `palm_other_ifls4` (all other palm farmer HHs). Runs separate and joint triple interactions (heat × IFLS5 × smallholder, heat × IFLS5 × other palm) to test whether the amplification is concentrated among smallholders who directly bear palm oil price risk. Panel sample with kecamatan/month/year/wave FE.

---

### Table N — `table_n_fuel_subsidy_card_mechanism.py`
**Gist:** Does the fuel-cut effect concentrate among urban-vehicle HHs that lacked a cash-transfer buffer (BLT/PSKS card)?

**Detail:** Constructs `no_fuel_buffer` = 1 if the household received neither cash transfer nor BLT card. Runs five specs: (1) headline triple reminder, (2) quadruple interaction adding no-buffer, (3) within urban-vehicle HHs: heat × post × no-buffer, (4) split sample: urban-vehicle + no buffer only → heat × post, (5) urban-vehicle + with buffer → heat × post. This tests whether government cash compensation buffered the mental-health impact of the fuel subsidy cut. IFLS5 only with kecamatan/month/year FE.

---

### Table O — `table_o_sw_three_stressors.py`
**Gist:** Subjective wellbeing channel — do the three stressors (palm, fuel, job loss) affect self-reported life satisfaction items?

**Detail:** Reads raw IFLS `b3a_sw.dta` for seven subjective wellbeing outcomes: Cantril ladder, living standard adequacy, food adequacy, health adequacy, family life adequacy, overall happiness, and likelihood of keeping living standard for 5 years. For each outcome, runs three regressions: palm shock (palm × IFLS5, panel), fuel shock (urban-vehicle × post-subsidy, IFLS5), and job loss (panel). This tests whether the stressors affect broader wellbeing, not just the clinical CES-D depression score.

---

### Table P — `table_p_jobloss_slope_controls.py`
**Gist:** Addresses slope confounding in the heat × job-loss interaction — adds heat-interacted controls for occupation, outdoor work, health, age, and income.

**Detail:** The concern is that job loss is endogenous: e.g., a construction labourer who lost a job differs from an office worker who didn't, and their heat sensitivity differs for reasons unrelated to unemployment. Sequentially adds: S1. heat × 9 IFLS4 sector dummies, S2. heat × outdoor work (agri/mining/construction), S3. heat × baseline health proxies, S4. heat × centered age, S5. heat × log PCE (cooling capacity), S6. all jointly. If the heat × job-loss coefficient survives, it's a real amplification effect; if it collapses, the baseline was a slope confound. Reads IFLS4 sector from raw `b3a_tk2.dta`. IFLS4-baseline panel with kecamatan/month/year/wave FE.

---

### Table Q — `table_q_palm_intensity_dose.py`
**Gist:** Dose-response — does the palm amplification scale with provincial palm acreage (continuous and tercile)?

**Detail:** Merges BPS provincial palm planted area (wave-matched: 2007 for IFLS4, 2014 for IFLS5) from `data/raw/palm_area_prov_BPS.csv`. Constructs a z-scored continuous intensity measure (`palm_int_z`, log of hectares) and tercile bins (low/mid/high acreage, plus zero for non-palm provinces). Runs: (1) binary palm farmer × IFLS5 (reference), (2) continuous intensity × IFLS5, (3) each tercile × IFLS5. If the effect scales with palm dependence and is absent where acreage is low, it's harder to attribute to "something else changed in 2014." Panel sample, kecamatan/month/year/wave FE.

---

### Table R — `table_r_palm_price_path.py`
**Gist:** Replaces the binary IFLS5 wave dummy with actual palm price at interview to see if the effect survives identification off the price itself.

**Detail:** IFLS5 fielding ran 2014-09 to 2015-09 while palm prices fell (715→511 USD/MT). Constructs five price encodings: (1) binary IFLS5 wave dummy (reference), (2) within-IFLS5 price decline (z-scored), (3) pooled cross-wave price level (no wave FE), (4) within-household price (pidlink FE), (5) IFLS4→IFLS5 household price gap. The honest finding noted in the docstring: the pooled price specs reproduce the effect, but within-IFLS5 and household price-gap variation is null — palm price is largely collinear with the wave, so price-timing alone doesn't separately identify it.

---

### Table S — `table_s_palm_placebo.py`
**Gist:** Placebo tests — crop placebo (actual cultivation) and fake-shock permutation placebo for the palm × heat amplification.

**Detail:** Panel A: runs the same heat × farmer × IFLS5 triple for seven actual crops (rice, maize, cassava, groundnuts, soybean, coffee, rubber) identified by IFLS cultivation codes. Each crop's 2007→2014 price change is annotated. Major staples (rice, maize) should be null if the effect is palm-specific. Panel B: randomly permutes palm-farmer status across households 200 times (seed 20240610) and re-estimates the triple each time; reports the true coefficient vs. the permutation null (5th/95th percentile, randomization-inference p-value). Uses 7-day mean heat, panel sample, kecamatan/month/year/wave FE.

---

### Table T — `table_t_crop_palm_overlap.py`
**Gist:** Diagnostic — are the crop placebo effects from Table S independent of palm, or just palm overlap?

**Detail:** Because "palm farmer" = agricultural HH × palm province (IFLS has no oil-palm crop code), a maize grower in a palm province is also flagged as a palm farmer. Panel A: tabulates crop-grower N, % in palm provinces, % also flagged as palm farmer. Panel B: runs the heat × grower × IFLS5 triple split by palm vs. non-palm provinces. Panel C: horse race — puts the crop triple and palm triple in the same model. If crop coefficients collapse while palm survives, the apparent crop effects are just palm overlap. Uses 7-day mean heat, panel sample, kecamatan/month/year/wave FE.

---

### Table U — `table_u_fuel_geography_robustness.py`
**Gist:** Fuel-cut geography robustness — FE ladder, pre/post balance, and placebo-in-time tests for the fuel triple.

**Detail:** Panel A: runs the heat × urban-vehicle × post-subsidy triple under a geography FE ladder (month+year → +province → +kabupaten → +kecamatan → +kabupaten×post → +kecamatan×post). The kecamatan×post FE absorbs any region-specific pre/post shift — if the triple survives, it's not a fielding-timing artifact. Panel B: pre vs. post-subsidy balance within IFLS5 on treatment, demographics, income, and heat (checks whether post tracks who was interviewed). Panel C: placebo-in-time — fake cutoff dates (Oct 1, 10, 20 2014) within the genuinely pre-hike window should yield no triple, arguing against seasonal confounds. IFLS5 only, kabupaten-clustered SE.

---

## Data Dependencies

Every table reads the canonical analysis input: `data/generated/30_analysis_table_input.parquet` (built by `uv run python code/data/main.py`).

Some tables additionally read raw IFLS sidecars or external reference files:

| Table | Extra input | Canonical path(s) |
|---|---|---|
| M — palm smallholder mechanism | IFLS `b3a_tk2.dta` (IFLS4 + IFLS5) | `data/raw/IFLS/extracted/IFLS4/hh07/b3a_tk2.dta`, `.../IFLS5/hh14/b3a_tk2.dta` |
| O — SW three stressors | IFLS `b3a_sw.dta` (IFLS4 + IFLS5) | `.../{IFLS4/hh07,IFLS5/hh14}/b3a_sw.dta` |
| P — job-loss slope controls | IFLS `b3a_tk2.dta` (IFLS4 only) | `data/raw/IFLS/extracted/IFLS4/hh07/b3a_tk2.dta` |
| Q — palm-intensity dose-response | BPS provincial palm-area CSV | `data/raw/palm_area_prov_BPS.csv` |
| R — palm price path | palm_price_usd_mt in canonical input | (already in parquet) |
| S — palm placebo | IFLS `b2_ut1.dta` (IFLS4 + IFLS5) | `.../{IFLS4/hh07,IFLS5/hh14}/b2_ut1.dta` |
| T — crop–palm overlap diagnostic | IFLS `b2_ut1.dta` (IFLS4 + IFLS5) | `.../{IFLS4/hh07,IFLS5/hh14}/b2_ut1.dta` |

Raw IFLS folders resolve from `code/data/config.py` (`IFLS4_FOLDER`, `IFLS5_FOLDER`), i.e. `data/raw/IFLS/extracted/IFLS4/hh07` and `.../IFLS5/hh14`.

## Notes for Reviewers

- **"Palm farmer"** = agricultural household × palm-producing province. IFLS has no oil-palm crop code, so palm cannot be identified by cultivation — hence the region × agriculture proxy. Other crops (rice, maize, rubber, coffee, ...) are identified by actual cultivation. Table T quantifies the resulting overlap and shows the crop placebos carry no effect independent of palm.
- Palm-identification tables (Q/R/S/T) use the 7-day mean temperature deviation (`tmean_7d_dev`) and kabupaten-clustered standard errors on the IFLS4-baseline panel.
