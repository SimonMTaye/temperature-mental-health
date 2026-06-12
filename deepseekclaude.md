# Analysis Tables — deepseekclaude reference

Each `table_<letter>_*.py` in `code/analysis/tables/` writes LaTeX and CSV to `output/tables/`. All read from `data/generated/30_analysis_table_input.parquet`; some additionally read raw IFLS sidecars. Shared helpers live in `_lettered_common.py`.

The paper studies how heat (temperature deviation) amplifies mental health (CES-D) responses, and whether four economic stressors — job loss, palm oil price collapse, coal price decline, and fuel subsidy cut — make households more heat-sensitive.

---

## Table A — Headline

**One-liner:** The five core specifications: baseline heat effect, and heat interacted with each of the four stressors (job loss, palm shock, coal shock, fuel subsidy cut).

**Detail:** Runs five OLS regressions of CES-D z-score on heat deviation (`heat_c_dev`), each with a different interaction term of interest. Column 1 is the main effect of heat alone. Column 2 interacts heat with a job-loss indicator (panel, pooled waves). Columns 3 and 4 interact heat with IFLS5 wave × a baseline-group dummy — palm farmer household and coal worker household respectively — identifying off the between-wave change. Column 5 is IFLS5-only: heat × post-subsidy-cut × urban vehicle household, identifying off interview timing relative to the November 2014 fuel hike. All use kecamatan FE, month/year FE, demographic controls, and kabupaten-clustered standard errors. The table extracts only the key interaction coefficient from each model.

---

## Table B — Day/Night Temperature

**One-liner:** Replicates the headline stressor interactions using Tmax and Tmin instead of the composite heat deviation, to check that results aren't driven by a particular temperature measure.

**Detail:** Takes the same four stressor interactions as Table A (job loss, palm shock, coal shock, fuel cut) and runs each with two alternative heat measures: Tmax deviation (`tmax_c_dev`, Panel A) and Tmin deviation (`tmin_c_dev`, Panel B). The structure is identical — same FEs, same controls, same clustering. The question is whether the amplification-by-stressor results are robust to using daytime highs vs nighttime lows vs the mean deviation. Each panel reports one row per heat-stressor interaction coefficient and its SE.

---

## Table C — Cooling Degree Days

**One-liner:** Replicates the stressor interactions using four cooling-degree-day (CDD) thresholds as the heat measure, to test sensitivity to alternative heat definitions that count extreme rather than average temperature.

**Detail:** Runs the same four stressor specifications (job loss, palm, coal, fuel) with four CDD variants: CDD above 30°C Tmax, CDD above 32°C Tmax, CDD above 23°C Tmin, and CDD above 24°C Tmin. CDD measures the cumulative degrees above a threshold rather than the mean deviation, so it captures the intensity and duration of extreme heat differently. Structure mirrors Table B exactly — one interaction coefficient per heat measure per stressor, with SEs underneath.

---

## Table D — Summary Statistics

**One-liner:** Reports means, standard deviations, and observation counts for every variable used in the analysis, organized by thematic panels.

**Detail:** Five panels: (A) mental health outcomes (CES-D z-score, raw score, depression indicator), (B) daily temperature exposure variables (mean, heat deviation, Tmax, Tmin), (C) stressors and baseline group indicators (job loss, palm farmer, coal worker, urban vehicle), (D) fuel-cut variables (post-subsidy, transport share, transport spending — IFLS5 only), and (E) demographic controls (age, female, education, married, widowed). Also reports the number of kabupaten clusters and kecamatan FE units. This is the "know your data" table.

---

## Table E — CES-D Factor Decomposition

**One-liner:** Re-runs the headline interactions on three CES-D subscales (somatic, depressed affect, positive affect) to see which dimension of mental health the heat-stressor amplification loads on.

**Detail:** The CES-D is decomposed into three z-scored factors: somatic/activity-related symptoms, depressed affect, and (reverse-coded) positive affect. Each serves as the dependent variable in the same four stressor specifications as Tables B/C (job loss, palm, coal, fuel). The structure is identical — one heat×stressor interaction coefficient per DV per stressor, with SEs. The table reveals whether the amplification is driven by somatic complaints (sleep, appetite), mood (sadness, loneliness), or the absence of positive feelings.

---

## Table F — Economic Mechanism

**One-liner:** Tests whether the palm, coal, and job-loss shocks predict changes in labor income, non-labor income, job loss, transport spending, and transport budget share — the economic channels through which stressors plausibly amplify heat sensitivity.

**Detail:** Four panels. Panel A regresses labor income, non-labor income, and job-loss probability on the palm shock (palm_farmer_hh × IFLS5) controlling for the palm-farmer main effect. Panel B does the same for coal. Panel C regresses labor and non-labor income on job_loss_1_yr. Panel D (IFLS5 only) regresses transport spending and transport budget share on the fuel shock (post_subsidy × urban_vehicle_hh). Each row reports a single coefficient, its SE, and N. The idea is to establish that these stressors meaningfully affect economic outcomes, making them plausible amplifiers of heat vulnerability.

---

## Table G — Sleep Mechanism

**One-liner:** Tests whether heat and heat×stressor interactions predict reduced sleep duration, establishing sleep as a candidate biological pathway.

**Detail:** IFLS5 only (sleep duration is only in wave 5). Five columns: heat alone, then heat interacted with each of the four stressors. Two rows of coefficients: the heat×stressor interaction (top) and the heat main effect (bottom), each with SEs. Sleep duration in hours is the dependent variable. The key idea is that heat disrupts sleep, and households under economic stress may be less able to buffer this disruption (no AC, worse housing, financial worry), making sleep loss a mechanism linking heat×stressor to mental health.

---

## Table H — Within-Day Hourly Temperature

**One-liner:** Replicates the headline results using the temperature deviation at the exact hour of the survey interview, with calendar-day FE to absorb day-level confounds.

**Detail:** Uses hourly temperature (`heat_hr_dev`) and hourly mean temperature (`tmean_c_hour`) and replaces the standard kecamatan + month + year FE structure with kecamatan + calendar-day FE (`day_id`). This means identification comes from variation in the interview-hour temperature across respondents interviewed on the same day in the same kecamatan. The five columns mirror Table A: heat main effect plus four stressor interactions. Singleton FE groups are dropped for both kecamatan and day_id dimensions.

---

## Table I — Heat Window Robustness

**One-liner:** Replaces the single-day heat deviation with three 7-day aggregate measures (mean temperature deviation, count of days above 30°C, and local heatwave days) to test that results aren't sensitive to the heat measurement window.

**Detail:** Same four stressor specifications as B/C/E, but the heat variables are now 7-day trailing windows: `tmean_7d_dev` (7-day mean temperature deviation from seasonal norm), `hot30_7d_dev` (count of days exceeding 30°C in the past week), and `heatwave_7d_dev` (local heatwave days in the past week). This addresses whether persistent heat exposure matters more than the single interview-day temperature. Each heat measure gets one row with the interaction coefficient across all four stressor columns.

---

## Table J — Job-Loss Recall Window

**One-liner:** Varies the recall window for job loss from 3 months to 60 months to see at what horizon the heat×job-loss interaction peaks, addressing recall bias and defining the relevant exposure window.

**Detail:** Runs eight versions of the heat×job-loss interaction, each using a different recall window: 90d, 180d, 270d, 365d, 540d, 730d, 1095d, and 1825d. Sample is restricted to the IFLS4-baseline panel. For each window, the table reports: number of treated households, share treated, the heat×job-loss interaction coefficient (with SE), and the job-loss main effect (with SE). This is about whether recent vs distant job loss matters differently for heat sensitivity.

---

## Table K — Job-Loss Reason Heterogeneity

**One-liner:** Splits job loss into involuntary (layoff, firm closure) vs family-related reasons, then estimates them jointly to test whether involuntary job loss is a stronger amplifier.

**Detail:** Panel A runs separate heat×job-loss regressions for three categories: any job loss, involuntary only, and family-related only. Each row reports n_treated, percent female among treated, the interaction coefficient, and the stressor main effect. Panel B puts involuntary and family-related loss in the same model and reports both coefficients plus their formal difference (linear combination test with clustered SE). Sample is IFLS4-baseline panel. The idea is that involuntary job loss is more exogenous and plausibly more stressful than quitting for family reasons.

---

## Table L — Temperature Balance Table

**One-liner:** A validity check: splits the sample at the median 7-day heat deviation and tests whether high-temp and low-temp interview days differ on observables (timing, demographics, fuel-cut variables).

**Detail:** Uses `tmean_7d_dev` (the identifying variation behind the fuel-cut result). Panel A checks whether interview month, day-of-year, and IFLS5 wave are balanced across high/low temp. Panel B checks demographics (age, female, education, marital status, household size, urban, log PCE, urban-vehicle status). Panel C (IFLS5 only) checks whether post-subsidy timing, cash-transfer receipt, and BLT card holding are balanced. Each variable reports low-mean, high-mean, the high-minus-low difference with significance stars and normalized difference, plus N. Inference is kabupaten-clustered. This is a standard balance table to support the claim that temperature variation is as-good-as-random conditional on FEs.

---

## Table M — Palm Smallholder vs Other-Palm Mechanism

**One-liner:** Decomposes the palm-farmer group into self-employed smallholders (who bear CPO price risk directly) vs wage/family/mixed workers, and tests whether the amplification is concentrated among smallholders.

**Detail:** Requires a raw IFLS sidecar (`b3a_tk2.dta`) to classify palm-farmer households at IFLS4 baseline by the household head's agricultural employment status. Pure smallholders (`palm_self_pure_ifls4`) are in palm provinces, in agriculture, self-employed, and not wage-employed. "Other palm" is everyone else flagged as a palm farmer. Two specifications: separate triples (heat × IFLS5 × group) and a joint model with both groups. If the effect loads on self-employed smallholders who actually sell palm oil (and thus bear the price decline), that strengthens the economic mechanism.

---

## Table N — Fuel Subsidy Card Mechanism

**One-liner:** Tests whether the fuel-subsidy-cut amplification is concentrated in urban vehicle households that did NOT have a cash-transfer buffer (BLT card or cash transfer), i.e., the uninsured.

**Detail:** IFLS5 only. Builds a `no_fuel_buffer` indicator (1 if the household received neither a cash transfer nor held a BLT/PSKS card at the time of the fuel hike). Runs five specifications: (1) the headline reminder triple; (2) a quadruple interaction adding the no-buffer flag; (3) within urban-vehicle households, a triple heat × post × no_buffer; (4) the heat × post interaction in the no-buffer urban-vehicle subsample only; (5) the same in the with-buffer urban-vehicle subsample. If the effect is driven by uninsured households, the no-buffer interaction should be large and the with-buffer one should be small/null.

---

## Table O — Subjective Wellbeing Channel

**One-liner:** Tests whether the palm shock, fuel shock, and job loss predict changes in seven subjective wellbeing items (Cantril ladder, living standards, food adequacy, health adequacy, family life, happiness, future expectations), establishing that these are genuinely stressful events.

**Detail:** Requires a raw IFLS sidecar (`b3a_sw.dta`) containing seven subjective wellbeing items measured at the individual level in both waves. Three columns: palm shock (palm_farmer_hh × IFLS5, panel), fuel shock (urban_vehicle × post_subsidy, IFLS5), and job loss (panel). Seven rows, one per SW item. Each cell reports the coefficient of the shock on that SW outcome (not interacted with heat — this is about the direct effect of the stressor on wellbeing). Codes 8 (don't know) and 9 (missing) are set to NaN. This validates that the "stressors" are indeed stressful.

---

## Table P — Job-Loss Slope Controls

**One-liner:** Adds heat-interacted controls (occupation sector, outdoor work, baseline health, age, log PCE) to the heat×job-loss specification to test whether the interaction survives — or is a proxy for differential heat sensitivity across types of workers.

**Detail:** IFLS4-baseline panel. Seven specifications stacking slope controls: S0 is the baseline; S1 adds heat interacted with 9 IFLS4 occupation sector dummies; S2 adds heat × outdoor work (agri/mining/construction); S3 adds heat × baseline health (many symptoms, recent hospitalization, recent accident); S4 adds heat × age (centered); S5 adds heat × log PCE (centered, as a cooling-capacity proxy); S6 adds all jointly. Each row reports the heat×job-loss coefficient. If the coefficient is stable across S1–S6, it's not a slope-confound artifact. There's a TODO to add AC ownership once it's cleaned.

---

## Table Q — Palm Intensity Dose-Response

**One-liner:** Replaces the binary palm-farmer indicator with continuous provincial palm acreage (from BPS statistics) and tercile bins to test whether the effect scales with palm dependence.

**Detail:** Merges provincial palm planted area (hectares, wave-matched 2007/2014) from an external BPS CSV. Builds: (1) a z-scored continuous palm intensity measure (`palm_int_z`), and (2) tercile bins (T1=low, T2=mid, T3=high acreage) plus a non-palm base (0). Runs four triples: the binary reference, the continuous dose, and the three tercile bins (from one joint model). Heat is `tmean_7d_dev`; kabupaten-clustered SE. If the coefficient increases monotonically with tercile, that's a dose-response gradient consistent with a causal palm story.

---

## Table R — Palm Price Path

**One-liner:** Replaces the binary IFLS5 wave dummy in the palm triple with the actual CPO price at each household's interview date, testing whether the effect is identifiable from price variation rather than just the between-wave change.

**Detail:** Uses `palm_price_usd_mt` and `palm_price_z` from the canonical input, plus a derived household-level IFLS4→IFLS5 price gap. Five specifications: (1) the binary IFLS5 dummy (reference); (2) palm price decline within IFLS5 (identifying off interview-timing variation); (3) price at interview pooled across waves with no wave FE (cross-wave price level); (4) same but with household FE (within-household price variation); (5) household IFLS4→IFLS5 price gap within IFLS5. The honest finding noted in the docstring is that the powered price-at-interview specs reproduce the effect, but purely within-wave price variation is null — price is largely collinear with the wave.

---

## Table S — Palm Placebo

**One-liner:** Two placebo tests: (A) runs the same wave-DiD triple on actual crop cultivation (rice, maize, cassava, etc.) to check whether the palm effect is crop-specific, and (B) a permutation test shuffling palm-farmer status to get a randomization-inference p-value.

**Detail:** Panel A requires raw IFLS `b2_ut1.dta` to identify crop growers by actual cultivation codes (ut07a/ut07b). For each crop with ≥60 growers, it runs heat × grower_ifls4 × IFLS5 and annotates the crop's real-world 2007-08 to 2014-15 price change. Palm (identified by region × agriculture, since IFLS has no oil-palm code) is the reference row. Panel B permutes palm-farmer status across households 200 times (with seed), fits the triple each time, and reports the true coefficient, the permutation null distribution (mean, 5th/95th percentiles), and the randomization-inference p-value. Null crops with null price changes that show no effect support the claim that palm is special.

---

## Table T — Crop-Palm Overlap Diagnostic

**One-liner:** Quantifies how much each crop-grower group overlaps with the palm-farmer proxy, and runs a horse race to see whether any crop effect survives controlling for palm.

**Detail:** Since "palm farmer" is proxied by agricultural HH × palm province (IFLS has no oil-palm crop code), a maize or coffee grower in a palm province is also flagged as a palm farmer. Panel A reports, for each crop, the number of IFLS4-baseline growers, the share in palm provinces, and the share also flagged as palm farmers. Panel B runs the crop triple (heat × grower × IFLS5) split by palm vs non-palm province. Panel C puts the crop triple and the palm triple in the same model — if the crop coefficient collapses while palm survives, the apparent crop effects are just palm overlap. This is a diagnostic table justifying the region×agriculture palm proxy.

---

## Table U — Fuel Geography Robustness

**One-liner:** Subjects the fuel-subsidy triple to a geography-FE ladder (up to kecamatan×post), a pre/post balance check, and a placebo-in-time test using fake cutoff dates in the pre-hike window.

**Detail:** IFLS5 only, heat = `tmean_7d_dev`. Panel A estimates the heat × urban_vehicle × post_subsidy triple under six increasingly stringent FE structures: month+year alone, then adding province, kabupaten, kecamatan, kabupaten×post, and kecamatan×post. The kecamatan×post FE absorbs any region-specific pre/post shift, so if the triple survives, it's not a fielding-timing confound. Panel B is a balance table on observables between pre- and post-subsidy interviews within IFLS5. Panel C assigns fake cutoff dates (Oct 1, Oct 10, Oct 20, 2014) inside the genuinely pre-hike window (before the real Nov 18, 2014 cut) and runs the triple; these should be null since no actual policy change occurred on those dates.
