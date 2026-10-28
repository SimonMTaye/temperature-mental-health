* mht_uncorrected.do - fit the 20 MHT models once with the baseline-kabupaten cluster and
* write .cache/stata_port/mht/uncorrected.csv (b, se, t, p, N, N_clust per hypothesis).
* p = 2 * ttail(df_r, |t|) with df_r = N_clust - 1, i.e. reghdfe's own clustered p-value.
*   /Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp -b do code/analysis/stata_v2/mht_uncorrected.do
version 19
clear all
set more off
adopath ++ ".cache/stata_port/ado"
include "code/analysis/stata_v2/mht_models.do"
capture mkdir ".cache"
capture mkdir ".cache/stata_port"
capture mkdir "$mht_dir"
log using "$mht_dir/uncorrected.log", text replace
mht_load
tempname post
postfile `post' str1 family int model str12 id str12 target double(b se t p N N_clust) using "$mht_dir/uncorrected.dta", replace
forvalues i = 1/$mht_n {
    mht_fit `i'
    local b = _b[${mht_target`i'}]
    local se = _se[${mht_target`i'}]
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    post `post' ("${mht_fam`i'}") (`i') ("${mht_id`i'}") ("${mht_target`i'}") (`b') (`se') (`t') (`p') (e(N)) (e(N_clust))
}
postclose `post'
use "$mht_dir/uncorrected.dta", clear
list, noobs
export delimited using "$mht_dir/uncorrected.csv", replace
log close
