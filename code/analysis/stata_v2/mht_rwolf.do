* mht_rwolf.do - Romano-Wolf (2005, 2016) step-down adjusted p-values via rwolf2 (Clarke),
* one family per run. Usage from the repository root (family = A or B):
*   stata-mp -b do code/analysis/stata_v2/mht_rwolf.do A 500
* Method: rwolf2 refits every model of the family on each cluster-bootstrap resample
* (bsample, cluster(kabupaten_full_code_ifls4) idcluster(bs_cluster); the regression's
* vce(cluster ...) is rewritten to the resampled cluster id), forms centred studentized
* statistics t*_j = (b*_j - b_j) / se*_j, and applies the step-down max-|t| algorithm:
* the largest observed |t| is compared with the distribution of max_j |t*_j| over the
* family, the next with the max over the remaining hypotheses, and so on, enforcing
* monotonicity. p = (#{max |t*| >= |t|} + 1) / (reps + 1). Also reports Holm (holm option).
* Caveat: duplicated clusters in a bootstrap sample share community and pidlink FE values,
* the usual limitation of the pairs cluster bootstrap with fixed effects.
version 19
clear all
set more off
args family reps
if "`family'" == "" local family "A"
if "`reps'" == "" local reps 500
local seed 20261009
adopath ++ ".cache/stata_port/ado"
include "code/analysis/stata_v2/mht_models.do"
capture mkdir "$mht_dir"
log using "$mht_dir/rwolf_`family'.log", text replace
which rwolf2
mht_load

local eqs
local vars
local idx
forvalues i = 1/$mht_n {
    if "${mht_fam`i'}" == "`family'" {
        local eqs `eqs' (${mht_cmd`i'})
        local vars = cond("`vars'" == "", "${mht_target`i'}", "`vars', ${mht_target`i'}")
        local idx `idx' `i'
    }
}
display `"Family `family': `: word count `idx'' hypotheses, `reps' resamples, seed `seed'"'
timer clear 1
timer on 1
rwolf2 `eqs', indepvars(`vars') cluster($mht_cluster) idcluster(bs_cluster) reps(`reps') seed(`seed') holm usevalid nodots
timer off 1
timer list 1
local secs = r(t1)
matrix RW = e(RW)

tempname post
postfile `post' str1 family int model str12 id str12 target double(p_unadj p_boot p_rwolf p_holm) int reps long seed double secs using "$mht_dir/rwolf_`family'.dta", replace
local j = 0
foreach i of local idx {
    local ++j
    post `post' ("`family'") (`i') ("${mht_id`i'}") ("${mht_target`i'}") (RW[`j', 1]) (RW[`j', 2]) (RW[`j', 3]) (RW[`j', 4]) (`reps') (`seed') (`secs')
}
postclose `post'
use "$mht_dir/rwolf_`family'.dta", clear
list, noobs
export delimited using "$mht_dir/rwolf_`family'.csv", replace
log close
