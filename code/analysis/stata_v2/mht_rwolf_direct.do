* mht_rwolf_direct.do - DEBUG: self-contained Romano-Wolf run with rwolf2, as a cross-check of
* mht_rwolf.do. Not part of the final output. Run from the repository root (family = A or B):
*   stata-mp -b do code/analysis/stata_v2/mht_rwolf_direct.do A 500
*   stata-mp -b do code/analysis/stata_v2/mht_rwolf_direct.do B 300
* mht_rwolf.do already calls rwolf2; the p-values are rwolf2's e(RW). What it adds is plumbing:
* the model registry and helper programs in mht_models.do. This file uses none of that: the
* reghdfe lines are written out literally from port_tables.do and passed straight to rwolf2, as
* in the rwolf2 help file. With the same seed, data order and commands it should reproduce
* rwolf_<family>.csv exactly; any difference would point to a plumbing bug.
* Two departures from port_tables.do are forced by rwolf2 itself, not by our code:
*   1. rwolf2 loops over the tested coefficients with `foreach var of varlist`, which rejects
*      factor-variable names (c.a#c.b#c.c), so the interactions are generated as product variables
*      below, in the same order reghdfe expands c.a##c.b##c.c;
*   2. idcluster() rewrites everything from "vce(" to the command's last ")", so vce() is last.
* Cluster: kabupaten_full_code_ifls4 (baseline kabupaten), as in mht_rwolf.do.
version 19
clear all
set more off
args family reps
if "`family'" == "" local family "A"
if "`reps'" == "" local reps = cond("`family'" == "A", 500, 300)
local seed 20261009
adopath ++ ".cache/stata_port/ado"
capture mkdir ".cache/stata_port/mht"
log using ".cache/stata_port/mht/rwolf_direct_`family'.log", text replace
which rwolf2

local controls "age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable"
local fe "stata_month_times_year stata_gadm_fullcode"
local opts "dof(clusters) tolerance(1e-12) vce(cluster kabupaten_full_code_ifls4)"

* Same columns, in the same row order, as mht_models.do's mht_load, so bsample draws match.
use stata_source_row pidlink stata_pidlink wave ifls5 post_subsidy gadm_fullcode stata_gadm_fullcode ///
    stata_month_times_year kabupaten_full_code kabupaten_full_code_ifls4 interview_date cesd_z ///
    palm_farmer_hh_ifls4 palm_price_gap_z urban_vehicle_hh_ifls4 urban_vehicle_transfer__0dc7241a ///
    urban_vehicle_transfer__60e2d6a2 job_loss_180d `controls' ///
    tmean_c tmax_c tmin_c heat_idx_c rh_pct tmean_7d tmin_7d tmean_30d wetbulb_c wetbulb_7d wetbulb_30d hot30_7d heatwave_7d ///
    using "data/generated/stata_port/analysis_port.dta", clear

if "`family'" == "A" {
    * Table A, IDs 1-6: CES-D on each heat measure (port_tables.do lines 36-61)
    rwolf2 ///
        (reghdfe cesd_z tmean_7d   `controls', absorb(`fe') `opts') ///
        (reghdfe cesd_z hot30_7d   `controls', absorb(`fe') `opts') ///
        (reghdfe cesd_z wetbulb_7d `controls', absorb(`fe') `opts') ///
        (reghdfe cesd_z tmean_c    `controls', absorb(`fe') `opts') ///
        (reghdfe cesd_z tmax_c     `controls', absorb(`fe') `opts') ///
        (reghdfe cesd_z wetbulb_c  `controls', absorb(`fe') `opts'), ///
        indepvars(tmean_7d, hot30_7d, wetbulb_7d, tmean_c, tmax_c, wetbulb_c) ///
        cluster(kabupaten_full_code_ifls4) idcluster(bs_cluster) reps(`reps') seed(`seed') holm usevalid
    local ids "table_a_1 table_a_2 table_a_3 table_a_4 table_a_5 table_a_6"
}

if "`family'" == "B" {
    * Table B, IDs 1-7 (7-day mean temperature, h = tm) and 8-14 (7-day wet-bulb, h = wb):
    * port_tables.do lines 98-172. Products of the shock group, post indicator and heat.
    gen double pf_i5  = palm_farmer_hh_ifls4 * ifls5
    gen double pp_i5  = palm_price_gap_z * ifls5
    gen double uv_ps  = urban_vehicle_hh_ifls4 * post_subsidy
    gen double uvn_ps = urban_vehicle_transfer__0dc7241a * post_subsidy
    gen double uvr_ps = urban_vehicle_transfer__60e2d6a2 * post_subsidy
    foreach h in tm wb {
        local heat = cond("`h'" == "tm", "tmean_7d", "wetbulb_7d")
        gen double i5_`h'     = ifls5 * `heat'
        gen double ps_`h'     = post_subsidy * `heat'
        gen double pf_`h'     = palm_farmer_hh_ifls4 * `heat'
        gen double pp_`h'     = palm_price_gap_z * `heat'
        gen double uv_`h'     = urban_vehicle_hh_ifls4 * `heat'
        gen double uvn_`h'    = urban_vehicle_transfer__0dc7241a * `heat'
        gen double uvr_`h'    = urban_vehicle_transfer__60e2d6a2 * `heat'
        gen double jl_`h'     = job_loss_180d * `heat'
        gen double pf_i5_`h'  = pf_i5 * `heat'
        gen double pp_i5_`h'  = pp_i5 * `heat'
        gen double uv_ps_`h'  = uv_ps * `heat'
        gen double uvn_ps_`h' = uvn_ps * `heat'
        gen double uvr_ps_`h' = uvr_ps * `heat'
    }
    local uvn urban_vehicle_transfer__0dc7241a
    local uvr urban_vehicle_transfer__60e2d6a2
    rwolf2 ///
        (reghdfe cesd_z palm_farmer_hh_ifls4 ifls5 pf_i5 tmean_7d pf_tm i5_tm pf_i5_tm `controls', absorb(`fe') `opts') ///
        (reghdfe cesd_z tmean_7d pf_i5_tm pf_tm i5_tm pf_i5 `controls', absorb(`fe' stata_pidlink ifls5) `opts') ///
        (reghdfe cesd_z palm_price_gap_z ifls5 pp_i5 tmean_7d pp_tm i5_tm pp_i5_tm `controls', absorb(`fe') `opts') ///
        (reghdfe cesd_z urban_vehicle_hh_ifls4 post_subsidy uv_ps tmean_7d uv_tm ps_tm uv_ps_tm `controls' if ifls5 == 1, absorb(`fe') `opts') ///
        (reghdfe cesd_z `uvn' post_subsidy uvn_ps tmean_7d uvn_tm ps_tm uvn_ps_tm `controls' if ifls5 == 1, absorb(`fe') `opts') ///
        (reghdfe cesd_z `uvr' post_subsidy uvr_ps tmean_7d uvr_tm ps_tm uvr_ps_tm `controls' if ifls5 == 1, absorb(`fe') `opts') ///
        (reghdfe cesd_z job_loss_180d tmean_7d jl_tm `controls', absorb(`fe') `opts') ///
        (reghdfe cesd_z palm_farmer_hh_ifls4 ifls5 pf_i5 wetbulb_7d pf_wb i5_wb pf_i5_wb `controls', absorb(`fe') `opts') ///
        (reghdfe cesd_z wetbulb_7d pf_i5_wb pf_wb i5_wb pf_i5 `controls', absorb(`fe' stata_pidlink ifls5) `opts') ///
        (reghdfe cesd_z palm_price_gap_z ifls5 pp_i5 wetbulb_7d pp_wb i5_wb pp_i5_wb `controls', absorb(`fe') `opts') ///
        (reghdfe cesd_z urban_vehicle_hh_ifls4 post_subsidy uv_ps wetbulb_7d uv_wb ps_wb uv_ps_wb `controls' if ifls5 == 1, absorb(`fe') `opts') ///
        (reghdfe cesd_z `uvn' post_subsidy uvn_ps wetbulb_7d uvn_wb ps_wb uvn_ps_wb `controls' if ifls5 == 1, absorb(`fe') `opts') ///
        (reghdfe cesd_z `uvr' post_subsidy uvr_ps wetbulb_7d uvr_wb ps_wb uvr_ps_wb `controls' if ifls5 == 1, absorb(`fe') `opts') ///
        (reghdfe cesd_z job_loss_180d wetbulb_7d jl_wb `controls', absorb(`fe') `opts'), ///
        indepvars(pf_i5_tm, pf_i5_tm, pp_i5_tm, uv_ps_tm, uvn_ps_tm, uvr_ps_tm, jl_tm, ///
                  pf_i5_wb, pf_i5_wb, pp_i5_wb, uv_ps_wb, uvn_ps_wb, uvr_ps_wb, jl_wb) ///
        cluster(kabupaten_full_code_ifls4) idcluster(bs_cluster) reps(`reps') seed(`seed') holm usevalid
    local ids
    forvalues j = 1/14 {
        local ids `ids' table_b_`j'
    }
}

* ---- e(RW) columns: model p, bootstrap p, Romano-Wolf p, Holm p (one row per hypothesis)
matrix RW = e(RW)
clear
svmat double RW
rename (RW1 RW2 RW3 RW4) (p_unadj p_boot p_rwolf p_holm)
gen str12 id = ""
forvalues j = 1/`: word count `ids'' {
    replace id = "`: word `j' of `ids''" in `j'
}
gen reps = `reps'
gen long seed = `seed'
order id
list, noobs
export delimited using ".cache/stata_port/mht/rwolf_direct_`family'.csv", replace
log close
