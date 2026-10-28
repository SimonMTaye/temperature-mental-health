* mht_qvalues.do - sharpened two-stage FDR q-values (Benjamini, Krieger & Yekutieli 2006)
* as implemented in Anderson (2008, JASA) "fdr_sharpened_qvalues.do", applied family by
* family to the uncorrected clustered p-values in .cache/stata_port/mht/uncorrected.csv
* (run mht_uncorrected.do first).
*   stata-mp -b do code/analysis/stata_v2/mht_qvalues.do
* Algorithm (Anderson's code, loop over q from 1 down to 0.001): for each candidate q,
* stage 1 runs BH at level q' = q/(1+q) and counts rejections r; stage 2 runs BH at level
* q' * M / (M - r); the q-value of hypothesis j is the smallest q at which it is rejected
* in stage 2. Correlations among p-values are ignored (valid under positive dependence).
version 19
clear all
set more off
include "code/analysis/stata_v2/mht_models.do"
log using "$mht_dir/qvalues.log", text replace

capture program drop mht_sharpened_qvalues
program define mht_sharpened_qvalues
    * Anderson (2008) sharpened q-values of variable `pval' into `qvar' for the data in memory.
    args pval qvar
    quietly {
        summarize `pval'
        local M = r(N)
        gen long _order = _n
        sort `pval'
        gen long _rank = _n if `pval' != .
        gen double `qvar' = 1 if `pval' != .
        local q = 1
        while `q' > 0 {
            local q1 = `q' / (1 + `q')
            gen double _crit1 = `q1' * _rank / `M'
            gen byte _rej1 = (_crit1 >= `pval') if `pval' != .
            gen double _rejrank1 = _rej1 * _rank
            egen double _nrej1 = max(_rejrank1)
            local q2 = `q1' * (`M' / (`M' - _nrej1[1]))
            gen double _crit2 = `q2' * _rank / `M'
            gen byte _rej2 = (_crit2 >= `pval') if `pval' != .
            gen double _rejrank2 = _rej2 * _rank
            egen double _nrej2 = max(_rejrank2)
            replace `qvar' = `q' if _rank <= _nrej2 & _rank != .
            drop _crit* _rej* _nrej*
            local q = round(`q' - .001, .001)
        }
        sort _order
        drop _order _rank
    }
end

import delimited using "$mht_dir/uncorrected.csv", clear case(preserve)
gen double q_sharpened = .
foreach fam in A B {
    preserve
    keep if family == "`fam'"
    mht_sharpened_qvalues p q_fam
    tempfile f
    save `f'
    restore
    merge 1:1 model using `f', keepusing(q_fam) nogenerate
    replace q_sharpened = q_fam if family == "`fam'"
    drop q_fam
}
sort model
format b se t p q_sharpened %20.15g
list family model id target p q_sharpened, noobs
export delimited using "$mht_dir/qvalues.csv", replace
log close
