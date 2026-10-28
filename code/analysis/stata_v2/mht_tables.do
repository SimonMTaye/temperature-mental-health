* mht_tables.do - refit the 20 MHT models (baseline-kabupaten cluster), attach the adjusted
* p-values written by mht_rwolf.do / mht_qvalues.do / mht_ri.do, and write one table per
* method to output/tables/stata_port/:
*   table_mht_rwolf.tex    p_unadj, p_rwolf (Romano-Wolf step-down), p_holm
*   table_mht_qvalues.tex  p_unadj, q_sharpened (Anderson 2008)
*   table_mht_ri.tex       p_unadj, p_ri (per-model RI), p_ri_fwer (family-wise, max-|t|)
* Each file holds a Family A panel (6 columns, coefficient renamed "heat") and a Family B
* panel (14 columns, coefficient renamed "diff_heat"); b and clustered se are shown.
*   stata-mp -b do code/analysis/stata_v2/mht_tables.do
version 19
clear all
set more off
adopath ++ ".cache/stata_port/ado"
include "code/analysis/stata_v2/mht_models.do"
capture mkdir "output/tables"
capture mkdir "$mht_out"
log using "$mht_dir/tables.log", text replace

* ---- adjusted p-values, one frame, one row per model
frame create adj
frame adj {
    import delimited using "$mht_dir/uncorrected.csv", clear case(preserve)
    keep model p
    rename p p_unadj
    foreach f in rwolf_A rwolf_B ri_A ri_B qvalues {
        capture confirm file "$mht_dir/`f'.csv"
        if _rc continue
        frame create tmp
        frame tmp: import delimited using "$mht_dir/`f'.csv", clear case(preserve)
        frlink 1:1 model, frame(tmp) generate(lnk)
        * Family A files create the columns; Family B files fill the remaining rows below.
        if "`f'" == "rwolf_A" frget p_rwolf p_holm, from(lnk)
        if "`f'" == "ri_A" frget p_ri p_ri_fwer, from(lnk)
        if "`f'" == "qvalues" frget q_sharpened, from(lnk)
        if "`f'" == "rwolf_B" {
            frget p_rwolf_b = p_rwolf, from(lnk)
            frget p_holm_b = p_holm, from(lnk)
            replace p_rwolf = p_rwolf_b if p_rwolf == .
            replace p_holm = p_holm_b if p_holm == .
            drop p_rwolf_b p_holm_b
        }
        if "`f'" == "ri_B" {
            frget p_ri_b = p_ri, from(lnk)
            frget p_ri_fwer_b = p_ri_fwer, from(lnk)
            replace p_ri = p_ri_b if p_ri == .
            replace p_ri_fwer = p_ri_fwer_b if p_ri_fwer == .
            drop p_ri_b p_ri_fwer_b
        }
        drop lnk
        frame drop tmp
    }
    list, noobs
}

* ---- refit, store, and attach scalars
mht_load
eststo clear
forvalues i = 1/$mht_n {
    mht_fit `i'
    foreach s in p_unadj p_rwolf p_holm q_sharpened p_ri p_ri_fwer {
        capture frame adj: confirm variable `s'
        if _rc continue
        frame adj: local v = `s'[`i']
        estadd scalar `s' = `v'
    }
    eststo ${mht_name`i'}
}

local A table_a_1 table_a_2 table_a_3 table_a_4 table_a_5 table_a_6
local B table_b_1 table_b_2 table_b_3 table_b_4 table_b_5 table_b_6 table_b_7 table_b_8 table_b_9 table_b_10 table_b_11 table_b_12 table_b_13 table_b_14
local renameA rename(tmean_7d heat hot30_7d heat wetbulb_7d heat tmean_c heat tmax_c heat wetbulb_c heat)
local renameB rename(pf4_i5_tm7 diff_heat ppg_i5_tm7 diff_heat uv_ps_tm7 diff_heat uvn_ps_tm7 diff_heat uvr_ps_tm7 diff_heat jl_tm7 diff_heat pf4_i5_wb7 diff_heat ppg_i5_wb7 diff_heat uv_ps_wb7 diff_heat uvn_ps_wb7 diff_heat uvr_ps_wb7 diff_heat jl_wb7 diff_heat)
local fmt se b(%9.4f) se(%9.4f) nonumbers mtitles nostar

capture program drop mht_write_table
program define mht_write_table
    args file stats labels
    local A table_a_1 table_a_2 table_a_3 table_a_4 table_a_5 table_a_6
    local B table_b_1 table_b_2 table_b_3 table_b_4 table_b_5 table_b_6 table_b_7 table_b_8 table_b_9 table_b_10 table_b_11 table_b_12 table_b_13 table_b_14
    local renameA rename(tmean_7d heat hot30_7d heat wetbulb_7d heat tmean_c heat tmax_c heat wetbulb_c heat)
    local renameB rename(pf4_i5_tm7 diff_heat ppg_i5_tm7 diff_heat uv_ps_tm7 diff_heat uvn_ps_tm7 diff_heat uvr_ps_tm7 diff_heat jl_tm7 diff_heat pf4_i5_wb7 diff_heat ppg_i5_wb7 diff_heat uv_ps_wb7 diff_heat uvn_ps_wb7 diff_heat uvr_ps_wb7 diff_heat jl_wb7 diff_heat)
    local fmt se b(%9.4f) se(%9.4f) nonumbers mtitles nostar
    local fmts %9.0fc %9.0f
    foreach s of local stats {
        local fmts `fmts' %9.4f
    }
    esttab `A' using "`file'", `renameA' keep(heat) stats(N N_clust `stats', fmt(`fmts') labels(N N_clust `labels')) `fmt' title("Family A: CES-D on heat measures (cluster: IFLS4 kabupaten)") replace
    esttab `B' using "`file'", `renameB' keep(diff_heat) stats(N N_clust `stats', fmt(`fmts') labels(N N_clust `labels')) `fmt' title("Family B: differential heat effect by shock (cluster: IFLS4 kabupaten)") append
end

mht_write_table "$mht_out/table_mht_rwolf.tex" "p_unadj p_rwolf p_holm" `""p (clustered)" "p Romano-Wolf" "p Holm""'
mht_write_table "$mht_out/table_mht_qvalues.tex" "p_unadj q_sharpened" `""p (clustered)" "q sharpened FDR""'
mht_write_table "$mht_out/table_mht_ri.tex" "p_unadj p_ri p_ri_fwer" `""p (clustered)" "p RI" "p RI family-wise""'
log close
