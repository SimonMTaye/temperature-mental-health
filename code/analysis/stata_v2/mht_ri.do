* mht_ri.do - randomisation inference with Alwyn Young's -randcmd- (SSC), one family per run.
*   stata-mp -b do code/analysis/stata_v2/mht_ri.do A 500
*   stata-mp -b do code/analysis/stata_v2/mht_ri.do B 500
* Design: the randomised unit is the community x calendar-date interview block
* (gadm_fullcode x interview_date; everyone interviewed in a community on a day shares one
* weather vector), passed to randcmd as groupvar(). Blocks are permuted within community x wave
* strata, strata(). ALL weather columns ($mht_weather) move together, so temperature and
* wet-bulb measures (used by different models of a family) are permuted jointly. Outcomes,
* controls, shock indicators, FE and the cluster variable are untouched. On every draw randcmd
* refits each model of the family (same cluster, same `if').
* How: randcmd's treatvar is an integer "source block" id (wx_src, initially = own block).
* randcmd permutes it across blocks within strata; calc1 (mht_ri_pull) then copies the whole
* weather vector of the source block into each row and rebuilds the heat x shock products. The
* weather columns are not passed as treatvars directly because randcmd's within-group constancy
* check is egen sd() > 0, and sd() of bit-identical doubles returns ~1e-15 rounding noise, so it
* rejects the design. An integer id has an exactly-zero sd.
* p-values written to ri_<family>.csv:
*   p_ri      = randcmd's randomization-t p (the "max-t" column of e(RCoef)):
*               (1 + #{p : |t*_ip| >= |t_i|}) / (P + 1)
*   p_ri_fwer = (1 + #{p : max_k |t*_kp| >= |t_i|}) / (P + 1), from the saved draws
*               (single-step Westfall-Young style max-|t|; randcmd's own e(RMult) only gives
*               one omnibus "any effect" p, not a per-hypothesis adjusted p)
* Ties use randcmd's convention: |t*| counts as >= |t| if |t*| > 0.999999 |t|.
* Tolerance: randcmd fits the actual and permuted models with the same command, so the equations
* use $mht_perm_opts (reghdfe default tolerance; see mht_models.do). The b, se, t and p_unadj
* written to ri_<family>.csv are refitted with the tight $mht_opts and match the port.
version 19
clear all
set more off
args family reps
if "`family'" == "" local family "A"
if "`reps'" == "" local reps 500
local seed 20261009
adopath ++ ".cache/stata_port/ado"
include "code/analysis/stata_v2/mht_models.do"
capture mkdir ".cache"
capture mkdir ".cache/stata_port"
capture mkdir "$mht_dir"
log using "$mht_dir/ri_`family'.log", text replace
which randcmd
set rng mt64
mht_load

* ---- models in this family
local models
forvalues i = 1/$mht_n {
    if "${mht_fam`i'}" == "`family'" local models `models' `i'
}
display "Family `family': models `models', `reps' permutations, seed `seed'"

* ---- randomisation blocks, strata, and the block-level weather lookup
egen long ri_block = group(gadm_fullcode wave interview_date)
egen long ri_stratum = group(gadm_fullcode wave)
gen long wx_src = ri_block
frame put ri_block $mht_weather, into(wx_blocks)
frame wx_blocks: bysort ri_block: keep if _n == 1

capture program drop mht_ri_pull
program define mht_ri_pull
    * After randcmd permutes wx_src: give each row the weather vector of its source block.
    capture drop wx_link
    drop $mht_weather
    quietly frlink m:1 wx_src, frame(wx_blocks ri_block) generate(wx_link)
    quietly frget $mht_weather, from(wx_link)
    quietly mht_build_interactions
end

* ---- randcmd equation list: ((target) reghdfe ..., $mht_perm_opts)
local eqs
foreach i of local models {
    local cmd `"${mht_cmd`i'}"'
    local cmd : subinstr local cmd "$mht_opts" "$mht_perm_opts"
    local eqs `"`eqs' ((${mht_target`i'}) `cmd')"'
}

timer clear 1
timer on 1
randcmd `eqs', treatvars(wx_src) strata(ri_stratum) groupvar(ri_block) reps(`reps') seed(`seed') ///
    saving("$mht_dir/ri_`family'_draws.dta", replace) calc1(mht_ri_pull)
timer off 1
quietly timer list 1
local secs = r(t1)
matrix R = e(RCoef)
matrix list R

* ---- actual estimates (tight tolerance) and randcmd's per-model p
local k = 0
foreach i of local models {
    local ++k
    mht_fit `i'
    local b`k' = _b[${mht_target`i'}]
    local se`k' = _se[${mht_target`i'}]
    local t`k' = abs(`b`k'' / `se`k'')
    local p_unadj`k' = 2 * ttail(e(df_r), `t`k'')
    local p_ri`k' = R[`k', 5]
}

* ---- family-wise p from the saved draws: compare each |t| with the max |t*| over the family
use "$mht_dir/ri_`family'_draws.dta", clear
local tstars
forvalues j = 1/`k' {
    gen double abst`j' = abs(ResB`j' / ResSE`j')
    local tstars `tstars' abst`j'
}
egen double maxt = rowmax(`tstars')
quietly count if maxt < .
local P = r(N)

tempname res
postfile `res' str1 family int model str12 id str12 target double(b se t p_unadj p_ri p_ri_fwer) int perms long seed double secs ///
    using "$mht_dir/ri_`family'.dta", replace
local j = 0
foreach i of local models {
    local ++j
    quietly count if maxt > 0.999999 * `t`j'' & maxt < .
    local p_fw = (1 + r(N)) / (`P' + 1)
    post `res' ("`family'") (`i') ("${mht_id`i'}") ("${mht_target`i'}") (`b`j'') (`se`j'') (`t`j'') (`p_unadj`j'') (`p_ri`j'') (`p_fw') (`P') (`seed') (`secs')
}
postclose `res'
use "$mht_dir/ri_`family'.dta", clear
list, noobs
export delimited using "$mht_dir/ri_`family'.csv", replace
log close
