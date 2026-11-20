"""Joint test that the three headline heat x stress interactions are all zero,
and precision-weighted (GLS) common interaction. 12 Nov 2026.

No new regressions. Inputs are the saved outputs of the R2.4 stacked equality
analysis (r2_comment4_current/equality/results): separate-model estimates/SEs
and the stacked district-clustered joint covariance. The stacked covariance uses
one global small-sample factor, which makes the individual-FE palm SE smaller
than in the source model. To stay conservative, the reported ("conservative")
covariance rescales the stacked correlation matrix by the separate-model SEs.
The unrescaled stacked covariance is also reported.
"""
import csv, json, math, pathlib
import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "r2_comment4_current" / "equality" / "results"
G = 289  # district clusters in the stack
out = []
for d in ["dry_sensitivity_panel_palm", "dry_primary_pooled_palm",
          "wet_sensitivity_panel_palm", "wet_primary_pooled_palm"]:
    rows = list(csv.DictReader(open(SRC / d / "coefficients.csv")))
    names = [r["design"] for r in rows]
    b = np.array([float(r["estimate"]) for r in rows])
    se_sep = np.array([float(r["se"]) for r in rows])
    Vs = np.array([[float(x) for x in r[1:]] for r in list(csv.reader(open(SRC / d / "joint_covariance.csv")))[1:]])
    sd = np.sqrt(np.diag(Vs)); R = Vs / np.outer(sd, sd)
    for label, V in [("conservative_separate_se", np.outer(se_sep, se_sep) * R), ("stacked_as_saved", Vs)]:
        def summarize(idx, tag):
            bb = b[idx]; VV = V[np.ix_(idx, idx)]; k = len(idx); Vi = np.linalg.inv(VV); one = np.ones(k)
            W = float(bb @ Vi @ bb)
            w = Vi @ one / (one @ Vi @ one); pooled = float(w @ bb); se = float((one @ Vi @ one) ** -0.5)
            avg = float(bb.mean()); se_avg = float(math.sqrt(one @ VV @ one) / k)
            tcrit = stats.t.ppf(0.975, G - 1)
            return dict(case=d, covariance=label, designs=tag, k=k,
                        wald_chi2=W, chi2_p=float(stats.chi2.sf(W, k)), F_p=float(stats.f.sf(W / k, k, G - 1)),
                        gls_pooled=pooled, gls_se=se, gls_ci_low=pooled - tcrit * se, gls_ci_high=pooled + tcrit * se,
                        gls_p=float(2 * stats.t.sf(abs(pooled / se), G - 1)), gls_weights="/".join(f"{x:.3f}" for x in w),
                        equal_weight_mean=avg, equal_weight_se=se_avg,
                        equal_weight_p=float(2 * stats.t.sf(abs(avg / se_avg), G - 1)))
        out.append(summarize([0, 1, 2], "+".join(names)))
        out.append(summarize([1, 2], "+".join(names[1:]) + " (palm excluded)"))
with open(HERE / "results.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(out[0].keys())); wr.writeheader(); wr.writerows(out)
for r in out:
    if r["covariance"] == "conservative_separate_se":
        print(f'{r["case"]:28s} {r["designs"]:42s} chi2({r["k"]})={r["wald_chi2"]:.2f} p={r["chi2_p"]:.4f} F_p={r["F_p"]:.4f} | GLS {r["gls_pooled"]:.4f} ({r["gls_se"]:.4f}) [{r["gls_ci_low"]:.3f},{r["gls_ci_high"]:.3f}] p={r["gls_p"]:.4f} w={r["gls_weights"]} | mean {r["equal_weight_mean"]:.4f} ({r["equal_weight_se"]:.4f}) p={r["equal_weight_p"]:.4f}')
