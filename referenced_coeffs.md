# Table numbers quoted in `main.tex`

Audit of every number in the prose of
`Dropbox/Apps/Overleaf/Temperature_Mental_Health_Paper/main.tex` (abstract, body,
footnotes, appendix notes) that comes from, or is computed from, a table read in
from `output_june_2026/tables/*.tex`. Commented-out lines (`%`) are skipped. Table
captions and notes contain no result numbers, only significance-star legends.
Two tables have no prose references: Table G (`tab:fuel-cash-transfer-het`, whose
`\input` is commented out at line 802) and the legacy `tab:cesd_items`
(`tables_figures/tables/table_cesd_items_body.tex`, whose `\input` is commented out
at line 725).

Cell names follow `table_<letter>_<stat>_c<col>_r<row>.tex`. Row numbering for the
tables cited here:
- **A, A2, E, I:** a single row, r1.
- **B, L, N, D, J, G:** B and L r1 = Differential (7-day temp), r2 = Heat (7-day
  temp), r3 = Differential (wet-bulb), r4 = Heat (wet-bulb); for L, r5/r6 is the
  positive-affect panel. N, D, J and G: r1 = Differential, r2 = Heat.
- **C:** r1 Age, r2 Female, r3 Education, r4 Married, r5 Widowed, r6 Cash transfer,
  r7 Work income, r8 Transport expenditure.
- **H:** r1 CES-D raw, r2 Depressed, r3 Daily mean temp, r4 7-day temp, r5 30-day
  temp, r6 7-day wet-bulb, r7 Job loss, r8 Palm farmers, r9 Urban vehicle owners,
  r10 Work income, r11 Nonfood exp., r12 Food exp.
- **Bottom-of-table numbers** keep counting after the coefficient rows, stat name
  `stat` (fixed-effect x/- rows are skipped). Fixed-effects tables (A, A2, D–G, I–K, M):
  custom rows (e.g. People with job loss, DV mean) first, then Observations, then R².
  B/L: r5 Observations, r6 Stressed Group Proportion (L: r7, r8). N: r3, r4.
  C: r9 Observations. H (in the SD column, c2): r13 Kabupaten clusters,
  r14 Kecamatan units, r15 Observations, r16 IFLS4, r17 IFLS5.

Match codes:
- **yes:** the paper prints the same value as the table.
- **rounding:** the paper's value is the table value rounded.
- **derived:** the paper's value is computed from table cells, and the formula
  reproduces it. Where it does not, the code is "derived – no".
- **no:** the value is wrong against the current table.

| main.tex line | quoted text | value in paper | table + cell (stat file) | current value in table .tex | match? |
|---|---|---|---|---|---|
| 48 | "robust negative impacts of heat ranging from 0.044" | 0.044 | B: table_b_coeff_c4_r1 (range min over binary cols 1,2,4,5,7) | 0.044* | yes |
| 48 | "to 0.09 SD per degree Celsius" | 0.09 | B: table_b_coeff_c2_r1 (range max) | 0.090*** | yes |
| 87 | "associated with a 0.001 standard-deviation change" | 0.001 | A: table_a_coeff_c1_r1 | 0.001 | yes |
| 87 | "similarly negligible coefficients of 0.006" | 0.006 | A: table_a_coeff_c4_r1 | 0.006 | yes |
| 87 | "of 0.006 and 0.004" | 0.004 | A: table_a_coeff_c5_r1 | 0.004 | yes |
| 87 | "rule out average effects larger than roughly 0.024" | 0.024 | A: table_a_coeff_c1_r1 + 1.96 × table_a_se_c1_r1 | 0.001 / (0.012) → 0.0245 | derived (printed values give 0.0245, so 0.024 or 0.025) |
| 89 | "raises depressive symptoms by 0.08 standard deviations" | 0.08 | B: table_b_coeff_c1_r1 | 0.080*** | yes |
| 89 | "after the fuel-subsidy cut the differential is 0.044" | 0.044 | B: table_b_coeff_c4_r1 | 0.044* | yes |
| 89 | "among the recently job-displaced it is 0.053" | 0.053 | B: table_b_coeff_c7_r1 | 0.053* | yes |
| 92 | "smallholders ... show a heat interaction of 0.073" | 0.073 | J: table_j_coeff_c2_r1 | 0.073*** | yes |
| 92 | "an effect that is about 10% larger" | 10% | J: table_j_coeff_c2_r1 / table_j_coeff_c3_r1 − 1 | 0.073*** / 0.065** → 12.3% | derived – no (~12%) |
| 240 | "pooled sample of n=58,906 individuals" | 58,906 | A: table_a_stat_c1_r2 (Observations) | 58,906 | yes (Table H, which the sentence cites, shows 59,146) |
| 240 | "IFLS-5-only sub-sample ... is n=20,038" | 20,038 | B: table_b_stat_c4_r5 … c6_r5 (Observations) | 20,038 | yes |
| 242 | "the mean CES-D-10 score is 5.3" | 5.3 | H: table_h_mean_c1_r1 | 5.283 | rounding |
| 242 | "15.6% of adults score at or above the ... cut-off" | 15.6% | H: table_h_mean_c1_r2 | 0.155 | no (15.5%) |
| 242 | "seven-day mean temperature ... averages 25.0°C" | 25.0 | H: table_h_mean_c1_r4 | 25.006 | rounding |
| 242 | "averages 25.0°C (SD 1.8)" | 1.8 | H: table_h_sd_c2_r4 | 1.815 | rounding |
| 242 | "almost identical to the single-day mean (25.0°C)" | 25.0 | H: table_h_mean_c1_r3 | 25.009 | rounding |
| 242 | "and the 30-day mean (25.0°C)" | 25.0 | H: table_h_mean_c1_r5 | 25.022 | rounding |
| 242 | "seven-day mean wet-bulb temperature averages 22.9°C" | 22.9 | H: table_h_mean_c1_r6 | 22.936 | rounding |
| 242 | "from 1.87 for the daily measure" | 1.87 | H: table_h_sd_c2_r3 | 1.857 | no (1.86) |
| 242 | "to 1.82 over seven days" | 1.82 | H: table_h_sd_c2_r4 | 1.815 | rounding |
| 242 | "and 1.78 over thirty" | 1.78 | H: table_h_sd_c2_r5 | 1.778 | rounding |
| 244 | "Recent job loss is rare, affecting 0.8% of adults" | 0.8% | H: table_h_mean_c1_r7 | 0.008 | yes |
| 244 | "palm-farmer households make up 12.2%" | 12.2% | H: table_h_mean_c1_r8 | 0.121 | no (12.1%; line 453 says 12.1%) |
| 244 | "urban vehicle-owning households 35.6%" | 35.6% | H: table_h_mean_c1_r9 | 0.357 | no (35.7%) |
| 244 | "The pooled sample contains 60,419 adults" | 60,419 | H: table_h_stat_c2_r15 (Observations) | 59,146 | no (line 240 also says 58,906) |
| 244 | "29,006 from IFLS-4" | 29,006 | H: table_h_stat_c2_r16 (IFLS4) | 28,662 | no |
| 244 | "and 31,413 from IFLS-5" | 31,413 | H: table_h_stat_c2_r17 (IFLS5) | 30,484 | no |
| 244 | "They span 2,014 kecamatans" | 2,014 | H: table_h_stat_c2_r14 (Kecamatan units) | 2,010 | no |
| 244 | "within 303 kabupatens" | 303 | H: table_h_stat_c2_r13 (Kabupaten clusters) | 303 | yes |
| 246 | "monthly employment earnings per household are \$179" | 179 | H: table_h_mean_c1_r10 | 178.57 | rounding |
| 246 | "monthly food expenditures ... are \$111" | 111 | H: table_h_mean_c1_r12 | 110.74 | rounding |
| 433 | "unrelated to depressive symptoms in 5 of the 6 specifications" | 5 of 6 | A: star count over table_a_pval_c1..c6_r1 | only c3 starred (-0.027*) | derived |
| 433 | "associated with a 0.001 standard-deviation change in CES-D" | 0.001 | A: table_a_coeff_c1_r1 | 0.001 | yes |
| 433 | "similarly small coefficients of 0.006" | 0.006 | A: table_a_coeff_c4_r1 | 0.006 | yes |
| 433 | "0.006, 0.004" | 0.004 | A: table_a_coeff_c5_r1 | 0.004 | yes |
| 433 | "and 0.009 (columns 4, 5, and 6)" | 0.009 | A: table_a_coeff_c6_r1 | 0.009 | yes |
| 433 | "insignificant coefficient of -0.004 (column 2)" | -0.004 | A: table_a_coeff_c2_r1 | -0.004 | yes |
| 433 | "rule out average effects larger than roughly 0.024" | 0.024 | A: table_a_coeff_c1_r1 + 1.96 × table_a_se_c1_r1 | 0.001 / (0.012) → 0.0245 | derived (printed values give 0.0245) |
| 435 | "marginally significant (column 3, -0.027" | -0.027 | A: table_a_coeff_c3_r1 | -0.027* | yes |
| 435 | "-0.027, p<0.10" | p<0.10 | A: table_a_pval_c3_r1 | -0.027* (SE 0.015, t≈1.8) | yes |
| 453 | "ten of the remaining twelve ... significant at the 5% level" | 10 of 12 | B: stars on table_b_pval_c{1,2,3,4,5,7}_r1 and _r3 | r1: \*\*\*,\*\*\*,\*\*\*,\*,\*\*,\*; r3: \*\*\*,\*\*\*,\*\*\*,\*\*,\*\*,\*\* | derived |
| 453 | "the remaining two are significant at the 10% level" | 2 | B: table_b_pval_c4_r1, table_b_pval_c7_r1 | 0.044*, 0.053* | derived |
| 453 | "range from 0.044 SD" | 0.044 | B: table_b_coeff_c4_r1 (min over r1, cols 1,2,4,5,7) | 0.044* | yes |
| 453 | "to 0.090 SD per degree Celsius" | 0.090 | B: table_b_coeff_c2_r1 (max) | 0.090*** | yes |
| 453 | "(0.067 SD ... for wet-bulb" | 0.067 | B: table_b_coeff_c7_r3 (min over r3) | 0.067** | yes |
| 453 | "to 0.094 SD for wet-bulb temperature" | 0.094 | B: table_b_coeff_c2_r3 (max) | 0.094*** | yes |
| 453 | "CES-D score that is 0.20" | 0.20 | B: 4.5 × table_b_coeff_c4_r1 | 4.5 × 0.044 = 0.198 | derived |
| 453 | "to 0.41 SD larger ... in the warm week" | 0.41 | B: 4.5 × table_b_coeff_c2_r1 | 4.5 × 0.090 = 0.405 | derived (rounding; the 4.5 is not in any table) |
| 453 | "12.1% of the population is impacted by the palm oil shock" | 12.1% | B: table_b_stat_c1_r6 (Stressed Group Proportion); also H: table_h_mean_c1_r8 | 0.121 | yes |
| 453 | "the point estimate of 0.08 standard deviations" | 0.08 | B: table_b_coeff_c1_r1 | 0.080*** | yes |
| 453 | "imply an overall effect of 0.01 standard deviations" | 0.01 | B: table_b_coeff_c1_r1 × Stressed Group Proportion (col 1) | 0.080 × 0.121 = 0.0097 | derived |
| 453 | "confidence interval ... ranges from -0.023" | -0.023 | A: table_a_coeff_c1_r1 − 1.96 × table_a_se_c1_r1 | 0.001 − 0.0235 = −0.0225 | derived |
| 453 | "-0.023 to 0.024 standard deviations" | 0.024 | A: table_a_coeff_c1_r1 + 1.96 × table_a_se_c1_r1 | 0.001 + 0.0235 = 0.0245 | derived (printed values give 0.0245) |
| 481 | "Vehicle owners after the subsidy cut are slightly younger (column 4)" | (sign/significance) | C: table_c_coeff_c4_r1 | -3.564** | yes |
| 481 | "all other demographics are well balanced" | (significance) | C: table_c_pval_c4_r3 | 0.607* (education) | no (education is significant at 10%) |
| 483 | "coefficients ranging from 0.078" | 0.078 | D: min of table_d_coeff_c1..c3_r1 | 0.080***, 0.080***, 0.081*** | no (min is 0.080) |
| 483 | "to 0.080 SD" | 0.080 | D: max of table_d_coeff_c1..c3_r1 (c3) | 0.081*** | no (max is 0.081) |
| 488 | "though about 10% larger for the self-employed" | 10% | J: table_j_coeff_c2_r1 / table_j_coeff_c3_r1 − 1 | 0.073*** / 0.065** → 12.3% | derived – no (~12%; with the paper's 0.067 it is ~9%) |
| 488 | "0.073 for self-employed smallholders (Column 2)" | 0.073 | J: table_j_coeff_c2_r1 | 0.073*** | yes |
| 488 | "and 0.067 for wage workers (Column 3)" | 0.067 | J: table_j_coeff_c3_r1 | 0.065** | no |
| 493 | "having lost a job from 246" | 246 | E: table_e_stat_c1_r2 (People with job loss) | 240 | no |
| 493 | "to 1,757" | 1,757 | E: table_e_stat_c6_r2 (People with job loss) | 1725 | no |
| 493 | "six- and nine-month windows (0.053" | 0.053 | E: table_e_coeff_c2_r1 | 0.053* | yes |
| 493 | "0.053 and 0.062 in Columns 2 and 3" | 0.062 | E: table_e_coeff_c3_r1 | 0.062** | yes |
| 493 | "the small number of very recent job-losers (246)" | 246 | E: table_e_stat_c1_r2 (People with job loss) | 240 | no |
| 502 | "statistically significant increase of \$4.59" | 4.59 | C: table_c_coeff_c4_r8 | 3.628 | no |
| 502 | "statistically significant increase" | (significance) | C: table_c_pval_c4_r8 | 3.628 (SE 2.294, no star, t≈1.58) | no (not significant) |
| 502 | "\$4.59 (25%) in transportation expenditures" | 25% | C: table_c_coeff_c4_r8 / table_c_mean_c1_r8 | 3.628 / 18.302 = 19.8% | derived – no (the old 4.59 / 18.302 = 25.1%) |
| 504 | "reduces monthly income by about USD 54" | 54 | I: table_i_coeff_c1_r1 | -54.227*** | yes |
| 504 | "a recent job loss by about USD 50" | 50 | I: table_i_coeff_c2_r1 | -50.287*** | yes |
| 504 | "equal to roughly 30%" | 30% | I: table_i_coeff_c1_r1 / DV mean | 54.227 / 178.57 = 30.4% | derived |
| 504 | "and 28% of the USD 179 mean" | 28% | I: table_i_coeff_c2_r1 / DV mean | 50.287 / 178.57 = 28.2% | derived |
| 504 | "of the USD 179 mean" | 179 | I: table_i_stat_c1_r2 (DV mean); also H: table_h_mean_c1_r10 | 178.57 | rounding |
| 504 | "both are significant at the 1% level" | p<0.01 | I: table_i_pval_c1_r1, table_i_pval_c2_r1 | \*\*\*, \*\*\* | yes |
| 517 | "heat significantly reduces sleep duration, by about five minutes (Column 1)" | ~5 min | N: table_n_coeff_c1_r1 × 60 | -0.084* h → 5.0 min | derived |
| 517 | "its modest magnitude (about five minutes)" | ~5 min | N: table_n_coeff_c1_r1 × 60 | -0.084* h → 5.0 min | derived |
| 556 | "depression symptoms between 0.044" | 0.044 | B: table_b_coeff_c4_r1 | 0.044* | yes |
| 556 | "and 0.090 SD per degree Celsius" | 0.090 | B: table_b_coeff_c2_r1 | 0.090*** | yes |

Lines 481 and 502 claim a significance level rather than quote a number. They are in
the table because they disagree with the current stars.

Text claims about significance checked with no mismatch:
- Line 92: the column 3 price-drop interaction is positive and significant (0.026***).
- Line 453: every differential is positive except column 6.
- Line 511: the sub-scale significance pattern matches Table L.
- Line 515: all six sleep coefficients in Table A2 are insignificant.
- Line 517: the job-loss sleep interaction is insignificant (0.074, SE 0.070).

## Unmatched

These numbers look like results but do not appear in any of the `\input` tables:

| main.tex line | quoted text | value | note |
|---|---|---|---|
| 179 | "with a 24-hour daily mean of 24.83°C" | 24.83 | It cites Figure `fig:temp`. Table H's daily mean temperature (table_h_mean_c1_r3) is 25.009, and line 242 says 25.0. Probably from an older sample. |
| 213 | "within which 73.6% of respondents were interviewed post-cut" | 73.6% | No table reports the post-cut share. |
| 453 | "10th percentile coolest week ... 4.5 degrees cooler than the 90th" | 4.5 | Not in any table. The commented line 461 gives p10 = 22.35 and p90 = 26.86 (difference 4.51), which is consistent. |
| 453 | "equivalent to 30% to 100% of the effect of direct interventions" | 30%–100% | 0.20–0.41 SD compared with cited literature effects, not with a table. |
| 502 | "corresponds closely to the 30% increase in fuel costs" | 30% | The policy fact from line 207, not a table. |
| 502 (fn) | "The remaining 5% gap" | 5% | Derived as 30% − 25%. It depends on the stale 25%: with the current Table C it would be about 10%. |

Data-description numbers not traced to tables (no check needed):
- 83% population coverage (lines 165, 556).
- About 30,000 individuals (lines 79, 165).
- Palm price USD 1,096 to USD 701/MT, a 36% decline (lines 191, 378). These come from the World Bank series and Figure `fig:palmprice`.
- Rp 211 trillion / USD 18 billion (line 207).
- Wet-bulb formula constants (line 177).
