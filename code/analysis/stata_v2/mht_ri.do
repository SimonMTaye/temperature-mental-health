* mht_ri.do - randomisation inference with date-block permutations, one family per run.
*   stata-mp -b do code/analysis/stata_v2/mht_ri.do B 500          (single run, writes ri_B.csv)
*   stata-mp -b do code/analysis/stata_v2/mht_ri.do B 125 <k>      (chunk k of the permutations)
* Design: the randomised unit is the community x calendar-date interview block
* (gadm_fullcode x interview_date; everyone interviewed in a community on a day shares one
* weather vector). Within each community x wave stratum the blocks' weather vectors are
* permuted among the stratum's dates, carrying ALL weather columns ($mht_weather) together,
* so temperature and wet-bulb measures (used by different models of a family) move jointly.
* Outcomes, controls, shock indicators, FE and the cluster variable are untouched.
* For each permutation every model in the family is refitted (same cluster, same `if`) and
* the studentized statistic |t*_j| = |b*_j / se*_j| of the target coefficient is recorded.
*   per-model RI p_j      = (1 + #{p : |t*_jp| >= |t_j|}) / (P + 1)
*   family-wise RI p_j    = (1 + #{p : max_k |t*_kp| >= |t_j|}) / (P + 1)   (single-step
*                           Westfall-Young style: compares with the max-|t| distribution)
* The p-values are computed by mht_ri_pvalues (mht_models.do), shared with mht_ri_combine.do.
* Chunked runs: with a third argument `chunk' the permutation loop runs with
* seed = 20261009 + chunk (base seed plus the chunk number, so chunks are reproducible and
* draw different permutations), writes ri_<family>_perms_<chunk>.dta, ri_<family>_checks_<chunk>.csv
* and ri_<family>_<chunk>.log, and stops without computing p-values. Run several chunks in
* parallel, then `mht_ri_combine.do <family>` appends the chunk files and writes ri_<family>.csv.
* Tolerance: the permuted refits use $mht_perm_opts (reghdfe default tolerance); the actual
* statistics use the tight $mht_opts. See the comment at mht_perm_opts in mht_models.do.
* Sanity checks for the first permutation are written to ri_<family>_checks[_<chunk>].csv: the
* correlation between permuted and actual tmean_7d, the within-block SD of the permuted series
* (must be 0), and the number of observations whose tmean_7d changed.
version 19
clear all
set more off
args family perms chunk
if "`family'" == "" local family "A"
if "`perms'" == "" local perms 500
local seed 20261009
local sfx
if "`chunk'" != "" {
    local seed = `seed' + `chunk'
    local sfx "_`chunk'"
}
adopath ++ ".cache/stata_port/ado"
include "code/analysis/stata_v2/mht_models.do"
capture mkdir "$mht_dir"
log using "$mht_dir/ri_`family'`sfx'.log", text replace
set rng mt64
set seed `seed'
mht_load

* ---- hypotheses in this family and the actual studentized statistics (tight tolerance)
mht_ri_actual `family'
local idx $mht_ri_idx
local tnames $mht_ri_tnames
local k : word count `idx'
display "Family `family': `k' hypotheses, `perms' permutations, seed `seed', chunk `chunk'"

* ---- block frame: one row per community x wave x date, original order k within stratum
foreach v of global mht_weather {
    gen double orig_`v' = `v'
}
frame put gadm_fullcode wave interview_date $mht_weather, into(blocks)
frame blocks {
    bysort gadm_fullcode wave interview_date: keep if _n == 1
    egen long stratum = group(gadm_fullcode wave)
    sort stratum interview_date
    by stratum: gen long k = _n
    count
    display "blocks: " r(N)
}
frame copy blocks source
frlink m:1 gadm_fullcode wave interview_date, frame(blocks) generate(blk)

capture program drop mht_permute_blocks
program define mht_permute_blocks
    * Shuffle weather vectors among the blocks of each stratum, then pull them into the main frame.
    frame blocks {
        foreach v in u pk src $mht_weather {
            capture drop `v'
        }
        gen double u = runiform()
        sort stratum u
        by stratum: gen long pk = _n
        frlink m:1 stratum pk, frame(source stratum k) generate(src)
        frget $mht_weather, from(src)
    }
    quietly frlink rebuild blk
    drop $mht_weather
    quietly frget $mht_weather, from(blk)
    quietly mht_build_interactions
end

local permsfile "$mht_dir/ri_`family'_perms`sfx'.dta"
tempname post
postfile `post' int perm double(`tnames') using "`permsfile'", replace
timer clear 1
timer on 1
forvalues p = 1/`perms' {
    mht_permute_blocks
    if `p' == 1 {
        quietly correlate tmean_7d orig_tmean_7d
        local corr = r(rho)
        quietly bysort gadm_fullcode wave interview_date: egen double sd_block = sd(tmean_7d)
        quietly summarize sd_block
        local maxsd = r(max)
        quietly count if tmean_7d != orig_tmean_7d
        local nchanged = r(N)
        quietly drop sd_block
        display "check perm 1: corr(perm, actual tmean_7d) = `corr'; max within-block sd = `maxsd'; obs changed = `nchanged' of " _N
        preserve
        clear
        set obs 1
        gen corr_tmean_7d = `corr'
        gen max_within_block_sd = `maxsd'
        gen obs_changed = `nchanged'
        export delimited using "$mht_dir/ri_`family'_checks`sfx'.csv", replace
        restore
    }
    local row
    foreach i of local idx {
        mht_fit `i' perm
        local tstar = abs(_b[${mht_target`i'}] / _se[${mht_target`i'}])
        local row `row' (`tstar')
    }
    post `post' (`p') `row'
    if mod(`p', 10) == 0 {
        timer off 1
        quietly timer list 1
        display "permutation `p' done, elapsed " r(t1) " s"
        timer on 1
    }
}
timer off 1
quietly timer list 1
local secs = r(t1)
postclose `post'
display "`perms' permutations in `secs' s"

if "`chunk'" != "" {
    * Chunk run: stamp the runtime and seed on the permutation file for mht_ri_combine.do and stop.
    use "`permsfile'", clear
    char _dta[secs] `secs'
    char _dta[seed] `seed'
    char _dta[chunk] `chunk'
    save "`permsfile'", replace
    log close
    exit
}

* ---- p-values (single, unchunked run)
mht_ri_pvalues `family' "`permsfile'" `seed' `secs'
log close
