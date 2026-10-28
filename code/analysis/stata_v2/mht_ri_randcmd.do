* mht_ri_randcmd.do - DEBUG: randomisation inference via Alwyn Young's -randcmd- (ssc), as a
* cross-check of the custom block-permutation loop in mht_ri.do. Not part of the final output.
*   stata-mp -b do code/analysis/stata_v2/mht_ri_randcmd.do <family> <reps> ["<model numbers>"]
*   e.g. mht_ri_randcmd.do A 500 1      (Table A, 7-day mean temperature only)
* Same design as mht_ri.do: the randomised unit is the community x interview-date block
* (groupvar), permuted within community x wave (strata), and all weather columns ($mht_weather)
* move together.
* How: randcmd's treatvar is an integer "source block" id (wx_src, initially = own block).
* randcmd permutes it across blocks within strata; calc1 (mht_randcmd_pull) then copies the
* whole weather vector of the source block into each row and rebuilds the heat x shock
* products, and every equation is refitted on that draw. The weather columns are not passed
* as treatvars directly because randcmd's within-group constancy check is egen sd() > 0, and
* sd() of bit-identical doubles returns ~1e-15 rounding noise (verified: 0 blocks have
* differing values, yet 23,104 rows get sd > 0), so it rejects the design. An integer id has
* an exactly-zero sd.
* Differences from mht_ri.do, immaterial for comparison: randcmd refits the actual model with
* the same command as the permutations, so the permuted-fit tolerance ($mht_perm_opts) is used
* throughout; the actual |t| differs from the tight-tolerance one in the 7th decimal or beyond.
* Outputs in $mht_dir: ri_randcmd_<family>.csv (per model: randcmd's own randomization-t p
* [max-t column of e(RCoef), = (1 + #{|t*| >= |t|}) / (P + 1), the same formula as mht_ri.do],
* plus per-test and family-wise p recomputed from the saved draws), ri_randcmd_<family>_draws.dta.
version 19
clear all
set more off
args family reps models
if "`family'" == "" local family "A"
if "`reps'" == "" local reps 500
local seed 20261009
adopath ++ ".cache/stata_port/ado"
include "code/analysis/stata_v2/mht_models.do"
capture mkdir "$mht_dir"
log using "$mht_dir/ri_randcmd_`family'.log", text replace
set rng mt64
mht_load
which randcmd

* ---- models in this run (default: the whole family)
if "`models'" == "" {
    forvalues i = 1/$mht_n {
        if "${mht_fam`i'}" == "`family'" local models `models' `i'
    }
}

* ---- randomisation blocks, strata, and the block-level weather lookup
egen long ri_block = group(gadm_fullcode wave interview_date)
egen long ri_stratum = group(gadm_fullcode wave)
gen long wx_src = ri_block
frame put ri_block $mht_weather, into(wx_blocks)
frame wx_blocks: bysort ri_block: keep if _n == 1

capture program drop mht_randcmd_pull
program define mht_randcmd_pull
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
display "models: `models'"

timer clear 1
timer on 1
randcmd `eqs', treatvars(wx_src) strata(ri_stratum) groupvar(ri_block) reps(`reps') seed(`seed') ///
    saving("$mht_dir/ri_randcmd_`family'_draws.dta", replace) calc1(mht_randcmd_pull)
timer off 1
quietly timer list 1
local secs = r(t1)
matrix R = e(RCoef)
matrix list R

* ---- actual |t| (tight tolerance, as in mht_ri.do) for the recomputed p-values
local k = 0
foreach i of local models {
    local ++k
    mht_fit `i'
    local t`k' = abs(_b[${mht_target`i'}] / _se[${mht_target`i'}])
    local p_randcmd`k' = R[`k', 5]
}

* ---- per-test and family-wise (max |t*|) p from the saved draws
use "$mht_dir/ri_randcmd_`family'_draws.dta", clear
local P = _N
local maxlist
forvalues j = 1/`k' {
    gen double abst`j' = abs(ResB`j' / ResSE`j')
    local maxlist `maxlist' abst`j'
}
egen double maxt = rowmax(`maxlist')

tempname res
postfile `res' str1 family int model str12 id str12 target double(t p_randcmd p_ri p_ri_fwer) int perms long seed double secs ///
    using "$mht_dir/ri_randcmd_`family'.dta", replace
local j = 0
foreach i of local models {
    local ++j
    quietly count if abst`j' >= `t`j''
    local p_ri = (1 + r(N)) / (`P' + 1)
    quietly count if maxt >= `t`j''
    local p_fw = (1 + r(N)) / (`P' + 1)
    post `res' ("`family'") (`i') ("${mht_id`i'}") ("${mht_target`i'}") (`t`j'') (`p_randcmd`j'') (`p_ri') (`p_fw') (`P') (`seed') (`secs')
}
postclose `res'
use "$mht_dir/ri_randcmd_`family'.dta", clear
list, noobs
export delimited using "$mht_dir/ri_randcmd_`family'.csv", replace
log close
