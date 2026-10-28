* mht_models.do - shared definitions for the multiple-hypothesis-testing (MHT) stage.
* Included (not run) by mht_rwolf.do, mht_ri.do, mht_qvalues.do and mht_tables.do so the
* 20 hypotheses are defined exactly once. Run everything from the repository root after
*   uv run python code/analysis/stata_v2/export_port_data.py
*
* Hypothesis families (paper tables):
*   Family A (6):  heat coefficient in table_a IDs 1-6  (CES-D on each heat measure).
*   Family B (14): differential-heat interaction in table_b IDs 1-14 (7 tmean_7d, 7 wetbulb_7d).
* Models are the reghdfe lines of port_tables.do with two changes:
*   1. Cluster = kabupaten_full_code_ifls4 (each person's IFLS4 kabupaten carried to both
*      waves; IFLS5-only entrants keep their own). 9.1% of panel individuals move kabupaten
*      between waves; clustering on the contemporaneous kabupaten would split one person's
*      two observations across independently resampled clusters (and break the pidlink-FE
*      model B/2 under resampling). vce() is deliberately the LAST option so rwolf2's
*      idcluster() regex rewrite only touches it.
*   2. Factor-variable products (c.a##c.b##c.c) are written as explicit product variables
*      built by mht_build_interactions, because rwolf2 reads targets as a plain varlist.
*      The expansions are numerically identical (verified against stage-1 coefficients).
* Identification assumption (unchanged): conditional on community FE, month*year FE and
* controls, interview-week weather is as good as random with respect to unobserved distress.

global mht_controls "age female edu_yrs_nullable married widowed divorced ethnicity_nullable religion_nullable"
global mht_fe "stata_month_times_year stata_gadm_fullcode"
global mht_cluster "kabupaten_full_code_ifls4"
global mht_opts "dof(clusters) tolerance(1e-12) vce(cluster $mht_cluster)"
* Permuted refits in mht_ri.do use reghdfe's default tolerance (1e-8) instead of 1e-12: the
* permutation distribution of |t*| is only used to count exceedances of the actual |t|, so
* the trailing digits of each permuted statistic are irrelevant, and the HDFE fixed-point
* iteration is the bulk of the per-fit cost. The actual (unpermuted) estimates reported in
* ri_<family>.csv always use $mht_opts so b/se/t still match the port to 15 digits.
global mht_perm_opts "dof(clusters) vce(cluster $mht_cluster)"
* Weather columns that are permuted jointly (as one date block) by the RI procedure.
global mht_weather "tmean_c tmax_c tmin_c heat_idx_c rh_pct tmean_7d tmin_7d tmean_30d wetbulb_c wetbulb_7d wetbulb_30d hot30_7d heatwave_7d"
global mht_groups "palm_farmer_hh_ifls4 palm_price_gap_z urban_vehicle_hh_ifls4 urban_vehicle_transfer__0dc7241a urban_vehicle_transfer__60e2d6a2 job_loss_180d"
global mht_keepvars "stata_source_row pidlink stata_pidlink wave ifls5 post_subsidy gadm_fullcode stata_gadm_fullcode stata_month_times_year kabupaten_full_code kabupaten_full_code_ifls4 interview_date cesd_z $mht_groups $mht_controls $mht_weather"
global mht_dir ".cache/stata_port/mht"
global mht_out "output/tables/stata_port"

* Load the slim dataset (only the columns the 20 models need) and build the products.
capture program drop mht_load
program define mht_load
    use $mht_keepvars using "data/generated/stata_port/analysis_port.dta", clear
    mht_build_interactions
end

* Explicit products. Re-run after permuting weather columns (RI) to refresh them.
* Suffixes: tm7 = tmean_7d, wb7 = wetbulb_7d; pf4 = palm_farmer_hh_ifls4, ppg = palm_price_gap_z,
* uv = urban_vehicle_hh_ifls4, uvn = ..._transfer_nonrecipient_ifls4 (_0dc7241a),
* uvr = ..._transfer_recipient_ifls4 (_60e2d6a2), jl = job_loss_180d, i5 = ifls5, ps = post_subsidy.
capture program drop mht_build_interactions
program define mht_build_interactions
    capture drop pf4_i5 ppg_i5 uv_ps uvn_ps uvr_ps
    gen double pf4_i5 = palm_farmer_hh_ifls4 * ifls5
    gen double ppg_i5 = palm_price_gap_z * ifls5
    gen double uv_ps  = urban_vehicle_hh_ifls4 * post_subsidy
    gen double uvn_ps = urban_vehicle_transfer__0dc7241a * post_subsidy
    gen double uvr_ps = urban_vehicle_transfer__60e2d6a2 * post_subsidy
    foreach w in tm7 wb7 {
        local src = cond("`w'" == "tm7", "tmean_7d", "wetbulb_7d")
        capture drop i5_`w' ps_`w' pf4_`w' ppg_`w' uv_`w' uvn_`w' uvr_`w' jl_`w' pf4_i5_`w' ppg_i5_`w' uv_ps_`w' uvn_ps_`w' uvr_ps_`w'
        gen double i5_`w'     = ifls5 * `src'
        gen double ps_`w'     = post_subsidy * `src'
        gen double pf4_`w'    = palm_farmer_hh_ifls4 * `src'
        gen double ppg_`w'    = palm_price_gap_z * `src'
        gen double uv_`w'     = urban_vehicle_hh_ifls4 * `src'
        gen double uvn_`w'    = urban_vehicle_transfer__0dc7241a * `src'
        gen double uvr_`w'    = urban_vehicle_transfer__60e2d6a2 * `src'
        gen double jl_`w'     = job_loss_180d * `src'
        gen double pf4_i5_`w' = pf4_i5 * `src'
        gen double ppg_i5_`w' = ppg_i5 * `src'
        gen double uv_ps_`w'  = uv_ps * `src'
        gen double uvn_ps_`w' = uvn_ps * `src'
        gen double uvr_ps_`w' = uvr_ps * `src'
    }
end

* Model registry: for i = 1..$mht_n, globals mht_fam`i' (A/B), mht_id`i' (paper table ID),
* mht_name`i' (eststo name), mht_target`i' (coefficient tested), mht_cmd`i' (full reghdfe call).
capture program drop mht_define_models
program define mht_define_models
    local i = 0
    * ---- Family A: table_a IDs 1-6 (port_tables.do: reghdfe cesd_z <heat> $controls, absorb($fe) $opts)
    foreach heat in tmean_7d hot30_7d wetbulb_7d tmean_c tmax_c wetbulb_c {
        local ++i
        global mht_fam`i' "A"
        global mht_id`i' "table_a_`i'"
        global mht_name`i' "table_a_`i'"
        global mht_target`i' "`heat'"
        global mht_cmd`i' "reghdfe cesd_z `heat' $mht_controls, absorb($mht_fe) $mht_opts"
    }
    * ---- Family B: table_b IDs 1-7 (tmean_7d) and 8-14 (wetbulb_7d)
    foreach w in tm7 wb7 {
        local src = cond("`w'" == "tm7", "tmean_7d", "wetbulb_7d")
        * B/1, B/8: palm_farmer_hh_ifls4 ## ifls5 ## heat
        local ++i
        global mht_fam`i' "B"
        global mht_id`i' "table_b_`=`i'-6'"
        global mht_name`i' "table_b_`=`i'-6'"
        global mht_target`i' "pf4_i5_`w'"
        global mht_cmd`i' "reghdfe cesd_z palm_farmer_hh_ifls4 ifls5 pf4_i5 `src' pf4_`w' i5_`w' pf4_i5_`w' $mht_controls, absorb($mht_fe) $mht_opts"
        * B/2, B/9: panel version with pidlink FE (wave FE absorbed)
        local ++i
        global mht_fam`i' "B"
        global mht_id`i' "table_b_`=`i'-6'"
        global mht_name`i' "table_b_`=`i'-6'"
        global mht_target`i' "pf4_i5_`w'"
        global mht_cmd`i' "reghdfe cesd_z `src' pf4_i5_`w' pf4_`w' i5_`w' pf4_i5 $mht_controls, absorb($mht_fe stata_pidlink ifls5) $mht_opts"
        * B/3, B/10: palm_price_gap_z ## ifls5 ## heat
        local ++i
        global mht_fam`i' "B"
        global mht_id`i' "table_b_`=`i'-6'"
        global mht_name`i' "table_b_`=`i'-6'"
        global mht_target`i' "ppg_i5_`w'"
        global mht_cmd`i' "reghdfe cesd_z palm_price_gap_z ifls5 ppg_i5 `src' ppg_`w' i5_`w' ppg_i5_`w' $mht_controls, absorb($mht_fe) $mht_opts"
        * B/4-6, B/11-13: urban vehicle groups ## post_subsidy ## heat, wave 5 only
        foreach g in uv uvn uvr {
            local gvar = cond("`g'" == "uv", "urban_vehicle_hh_ifls4", cond("`g'" == "uvn", "urban_vehicle_transfer__0dc7241a", "urban_vehicle_transfer__60e2d6a2"))
            local ++i
            global mht_fam`i' "B"
            global mht_id`i' "table_b_`=`i'-6'"
            global mht_name`i' "table_b_`=`i'-6'"
            global mht_target`i' "`g'_ps_`w'"
            global mht_cmd`i' "reghdfe cesd_z `gvar' post_subsidy `g'_ps `src' `g'_`w' ps_`w' `g'_ps_`w' $mht_controls if ifls5 == 1, absorb($mht_fe) $mht_opts"
        }
        * B/7, B/14: job_loss_180d ## heat
        local ++i
        global mht_fam`i' "B"
        global mht_id`i' "table_b_`=`i'-6'"
        global mht_name`i' "table_b_`=`i'-6'"
        global mht_target`i' "jl_`w'"
        global mht_cmd`i' "reghdfe cesd_z job_loss_180d `src' jl_`w' $mht_controls, absorb($mht_fe) $mht_opts"
    }
    global mht_n = `i'
end
mht_define_models

* Fit model i (quietly) and leave the estimate in e(); callers read _b/_se of $mht_target`i'.
* `mht_fit i perm` swaps $mht_opts for $mht_perm_opts (looser tolerance, permuted refits only).
capture program drop mht_fit
program define mht_fit
    args i mode
    local cmd `"${mht_cmd`i'}"'
    if "`mode'" == "perm" local cmd : subinstr local cmd "$mht_opts" "$mht_perm_opts"
    quietly `cmd'
end

* ---- Randomisation-inference helpers shared by mht_ri.do and mht_ri_combine.do.
* mht_ri_actual `family': fits every model of the family with the tight $mht_opts and stores
* globals mht_ri_idx (model numbers), mht_ri_tnames (t<i> column names) and, per model i,
* mht_ri_b`i', mht_ri_se`i', mht_ri_t`i' (= |t|) and mht_ri_p`i' (clustered two-sided p).
capture program drop mht_ri_actual
program define mht_ri_actual
    args family
    global mht_ri_idx
    global mht_ri_tnames
    forvalues i = 1/$mht_n {
        if "${mht_fam`i'}" != "`family'" continue
        mht_fit `i'
        global mht_ri_b`i' = _b[${mht_target`i'}]
        global mht_ri_se`i' = _se[${mht_target`i'}]
        global mht_ri_t`i' = abs(${mht_ri_b`i'} / ${mht_ri_se`i'})
        global mht_ri_p`i' = 2 * ttail(e(df_r), ${mht_ri_t`i'})
        global mht_ri_idx $mht_ri_idx `i'
        global mht_ri_tnames $mht_ri_tnames t`i'
    }
end

* mht_ri_pvalues `family' `permsfile' `seed' `secs': reads the permutation table (one row per
* permutation, columns t<i> = |t*_i|), sets P = number of rows and writes ri_<family>.dta/.csv:
*   p_ri      = (1 + #{p : |t*_ip| >= |t_i|}) / (P + 1)
*   p_ri_fwer = (1 + #{p : max_k |t*_kp| >= |t_i|}) / (P + 1)   (single-step max-|t|)
* Requires mht_ri_actual to have run first. Replaces the data in memory.
capture program drop mht_ri_pvalues
program define mht_ri_pvalues
    args family permsfile seed secs
    use "`permsfile'", clear
    local perms = _N
    egen double tmax = rowmax($mht_ri_tnames)
    tempname res
    postfile `res' str1 family int model str12 id str12 target double(b se t p_unadj p_ri p_ri_fwer) int perms long seed double secs using "$mht_dir/ri_`family'.dta", replace
    foreach i of global mht_ri_idx {
        quietly count if t`i' >= ${mht_ri_t`i'}
        local p_ri = (1 + r(N)) / (`perms' + 1)
        quietly count if tmax >= ${mht_ri_t`i'}
        local p_fw = (1 + r(N)) / (`perms' + 1)
        post `res' ("`family'") (`i') ("${mht_id`i'}") ("${mht_target`i'}") (${mht_ri_b`i'}) (${mht_ri_se`i'}) (${mht_ri_t`i'}) (${mht_ri_p`i'}) (`p_ri') (`p_fw') (`perms') (`seed') (`secs')
    }
    postclose `res'
    use "$mht_dir/ri_`family'.dta", clear
    list, noobs
    export delimited using "$mht_dir/ri_`family'.csv", replace
end
