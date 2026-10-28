* mht_ri_combine.do - combine chunked randomisation-inference runs of mht_ri.do into ri_<family>.csv.
*   stata-mp -b do code/analysis/stata_v2/mht_ri_combine.do B
* Appends every ri_<family>_perms_<chunk>.dta (and the unchunked ri_<family>_perms.dta if it
* exists) into ri_<family>_allperms.dta, refits the actual statistics once with the tight
* $mht_opts (mht_ri_actual) and computes p_ri / p_ri_fwer with P = total number of appended
* permutations (mht_ri_pvalues), writing ri_<family>.dta/.csv in the same format as a single
* mht_ri.do run so mht_tables.do is unchanged. The csv's `seed' column holds the base seed
* 20261009 (chunk k used 20261009 + k); `secs' is the sum of the chunks' loop runtimes.
version 19
clear all
set more off
args family
if "`family'" == "" local family "A"
local seed 20261009
adopath ++ ".cache/stata_port/ado"
include "code/analysis/stata_v2/mht_models.do"
log using "$mht_dir/ri_`family'_combine.log", text replace

* ---- permutation files, in name order
local files : dir "$mht_dir" files "ri_`family'_perms_*.dta"
local files : list sort files
local files : list clean files
capture confirm file "$mht_dir/ri_`family'_perms.dta"
if _rc == 0 local files `files' ri_`family'_perms.dta
display "combining: `files'"
local secs 0
tempfile all
local first 1
foreach f of local files {
    use "$mht_dir/`f'", clear
    local chunk : char _dta[chunk]
    local fsecs : char _dta[secs]
    if "`chunk'" == "" local chunk 0
    if "`fsecs'" == "" local fsecs 0
    gen int chunk = `chunk'
    local secs = `secs' + `fsecs'
    display "`f': " _N " permutations, chunk `chunk', seed " cond("`: char _dta[seed]'" == "", "unknown", "`: char _dta[seed]'") ", `fsecs' s"
    if `first' save "`all'", replace
    else {
        append using "`all'"
        save "`all'", replace
    }
    local first 0
}
use "`all'", clear
sort chunk perm
replace perm = _n
save "$mht_dir/ri_`family'_allperms.dta", replace
display "total permutations: " _N ", total loop time `secs' s"

* ---- actual statistics (tight tolerance) and p-values
mht_load
mht_ri_actual `family'
mht_ri_pvalues `family' "$mht_dir/ri_`family'_allperms.dta" `seed' `secs'
log close
