import sys, subprocess, os, pandas as pd
B, W, O = sys.argv[1], sys.argv[2], sys.argv[3]
def st(p): return "***" if p<.01 else "**" if p<.05 else "*" if p<.10 else ""
def f(x,d=3): return ("$-$" if round(x,d)<0 else "")+f"{abs(x):.{d}f}"
def doc(tab, title, body, notes, landscape=False, size=r"\small"):
    geo = "a4paper,landscape,margin=0.7in" if landscape else "a4paper,margin=0.72in"
    return rf"""\documentclass[11pt]{{article}}
\usepackage[{geo}]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern,booktabs,caption}}
\captionsetup[table]{{labelfont=bf,skip=12pt}}
\renewcommand{{\thetable}}{{{tab}}}
\begin{{document}}
\thispagestyle{{empty}}
\begin{{table}}[!htbp]\centering
\caption{{{title}}}
{size}
\renewcommand{{\arraystretch}}{{1.15}}
\setlength{{\tabcolsep}}{{3pt}}
{body}
\vspace{{9pt}}

\begin{{minipage}}{{\textwidth}}\footnotesize
\textit{{Notes:}} {notes}
\end{{minipage}}
\end{{table}}
\end{{document}}
"""
STD = r"District-clustered standard errors appear in parentheses. $^*p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$ (unadjusted tests)."
out = {}
# ---------- R3.3 both sides
r = pd.read_csv(f"{W}/fuel_both_sides_results.csv")
order = ["urban_vehicle_hh_ifls4", "urban_vehicle_transfer_nonrecipient_ifls4", "urban_vehicle_transfer_recipient_ifls4"]
def line(h):
    x = pd.concat([r[(r.heat==h)&(r.group==g)&(r["sample"]==s)] for g in order for s in ["full","both_sides"]])
    return " & ".join(f(e)+st(p) for e,p in zip(x.estimate,x.p)), " & ".join(f"({s:.3f})" for s in x.se)
a, b = line("tmean_7d"), line("wetbulb_7d")
body = rf"""\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lcccccc@{{}}}}
\toprule
 & \multicolumn{{6}}{{c}}{{Dependent variable: CES-D z-score}} \\
\cmidrule(lr){{2-7}}
 & \multicolumn{{2}}{{c}}{{\shortstack{{Urban vehicle\\owners}}}} & \multicolumn{{2}}{{c}}{{\shortstack{{Urban vehicle owners:\\no cash transfer}}}} & \multicolumn{{2}}{{c}}{{\shortstack{{Urban vehicle owners:\\cash transfer}}}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}
 & Full & Both sides & Full & Both sides & Full & Both sides \\
 & (1) & (2) & (3) & (4) & (5) & (6) \\
\midrule
\multicolumn{{7}}{{l}}{{\textit{{Panel A. Mean temperature}}}} \\
\addlinespace[4pt]
Group $\times$ post $\times T$ & {a[0]} \\
 & {a[1]} \\
\addlinespace[8pt]
\multicolumn{{7}}{{l}}{{\textit{{Panel B. Wet-bulb temperature}}}} \\
\addlinespace[4pt]
Group $\times$ post $\times T$ & {b[0]} \\
 & {b[1]} \\
\addlinespace[8pt]
\midrule
Kecamatan FE & Yes & Yes & Yes & Yes & Yes & Yes \\
Month-year FE & Yes & Yes & Yes & Yes & Yes & Yes \\
Demographic controls & Yes & Yes & Yes & Yes & Yes & Yes \\
\midrule
Observations & 20,038 & 6,922 & 20,038 & 6,922 & 20,038 & 6,922 \\
\bottomrule
\end{{tabular*}}"""
notes = r"""Each column reports the coefficient on the group named in the column heading interacted with the post-cut indicator and $T$, the seven-day average temperature measure in the panel, in $^\circ$C. The post-cut indicator equals one for interviews on or after 18 November 2014. Coefficients are CES-D standard deviations per $1^\circ$C. ``Full'' columns use all IFLS5 respondents and reproduce the paper. ``Both sides'' columns keep only kecamatan with at least one IFLS5 interview before and one after the subsidy cut (216 kecamatan, which contain 91\% of respondents interviewed before the cut).

\smallskip
Groups are defined from IFLS4 urban residence and vehicle ownership. All models include the demographic controls (age, sex, education, marital status, ethnicity and religion) and all lower-order terms. """ + STD
out["r3_3_both_sides"] = doc("R3.3", "Fuel-Subsidy Cut: Kecamatan Interviewed Before and After the Cut", body, notes)
# ---------- R3.9
g = pd.read_csv(f"{B}/remaining_review_batch/fuel_and_geography/results/geography_comparison.csv")
def gl(h, pre):
    x = g[g.heat==h].sort_values("column")
    return " & ".join(f(e)+st(p) for e,p in zip(x[pre+"_estimate"], x[pre+"_p"])), " & ".join(f"({s:.3f})" for s in x[pre+"_se"])
rows=[]
for h,lab in [("tmean_7d","Panel A. Mean temperature"),("wetbulb_7d","Panel B. Wet-bulb temperature")]:
    s1, s2 = gl(h,"source"), gl(h,"restricted")
    rows.append(rf"""\multicolumn{{8}}{{l}}{{\textit{{{lab}}}}} \\
\addlinespace[4pt]
All observations & {s1[0]} \\
 & {s1[1]} \\
\addlinespace[3pt]
Kecamatan-level temperature only & {s2[0]} \\
 & {s2[1]} \\
\addlinespace[8pt]""")
x = g[g.heat=="tmean_7d"].sort_values("column")
body = rf"""\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lccccccc@{{}}}}
\toprule
 & \multicolumn{{7}}{{c}}{{Dependent variable: CES-D z-score}} \\
\cmidrule(lr){{2-8}}
 & \shortstack{{Palm\\proxy}} & \shortstack{{Palm\\panel}} & \shortstack{{Palm\\price gap}} & \shortstack{{Urban vehicle\\owners}} & \shortstack{{Urban vehicle\\owners: no\\cash transfer}} & \shortstack{{Urban vehicle\\owners: cash\\transfer}} & \shortstack{{Job\\loss}} \\
 & (1) & (2) & (3) & (4) & (5) & (6) & (7) \\
\midrule
{chr(10).join(rows)}
\midrule
Observations, all & {" & ".join(f"{int(v):,}" for v in x.source_n)} \\
Observations, kecamatan-level only & {" & ".join(f"{int(v):,}" for v in x.restricted_n)} \\
\bottomrule
\end{{tabular*}}"""
notes = r"""Each column is one specification from the paper's main table. The coefficient shown is the heat $\times$ stressor interaction: group $\times$ IFLS5 $\times T$ in columns (1)--(3), group $\times$ post-cut $\times T$ in columns (4)--(6), and job loss $\times T$ in column (7). $T$ is the seven-day average temperature measure in the panel, in $^\circ$C. ``All observations'' reproduces the paper. ``Kecamatan-level temperature only'' drops the 1.05\% of observations whose location could be matched only to a kabupaten or province, and whose temperature is therefore measured at that coarser level. Column (2) includes individual fixed effects; when one observation of a respondent is dropped, the other is dropped as well. All models include the paper's demographic controls and kecamatan and month-year fixed effects. """ + STD
out["r3_9_temperature_assignment"] = doc("R3.9", "Robustness to Excluding Coarsely Assigned Temperature", body, notes, landscape=True)
# ---------- R3.10
m = pd.read_csv(f"{B}/remaining_review_batch/inference/results/models.csv"); m = m[m.model_id.str.startswith("palm_")]
def col(h):
    res = []
    for c in ["palm_col1","palm_col2","palm_col3"]:
        x = m[m.model_id==f"{c}_{h}"].reset_index(drop=True); res.append(x)
    est = " & ".join(f(x.estimate[0]) for x in res)
    L = [f"Palm exposure $\\times$ IFLS5 $\\times T$ & {est} \\\\", r"\addlinespace[3pt]"]
    for i,lab in enumerate(["District (paper)","District and year-month (two-way)","Baseline province"]):
        L.append(f"\\quad SE clustered by: {lab} & " + " & ".join(f"({x.se[i]:.3f}){st(x.p[i])}" for x in res) + r" \\")
    return "\n".join(L)
body = rf"""\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lccc@{{}}}}
\toprule
 & \multicolumn{{3}}{{c}}{{Dependent variable: CES-D z-score}} \\
\cmidrule(lr){{2-4}}
 & \shortstack{{Palm proxy\\(1)}} & \shortstack{{Palm panel\\(2)}} & \shortstack{{Palm price gap\\(3)}} \\
\midrule
\multicolumn{{4}}{{l}}{{\textit{{Panel A. Mean temperature}}}} \\
\addlinespace[4pt]
{col("tmean_7d")}
\addlinespace[8pt]
\multicolumn{{4}}{{l}}{{\textit{{Panel B. Wet-bulb temperature}}}} \\
\addlinespace[4pt]
{col("wetbulb_7d")}
\addlinespace[8pt]
\midrule
Individual FE & No & Yes & No \\
Kecamatan FE & Yes & Yes & Yes \\
Month-year FE & Yes & Yes & Yes \\
Demographic controls & Yes & Yes & Yes \\
\midrule
Observations & 48,824 & 40,012 & 41,152 \\
District clusters & 277 & 273 & 273 \\
Year-month clusters & 23 & 22 & 22 \\
Baseline-province clusters & 20 & 20 & 20 \\
\bottomrule
\end{{tabular*}}"""
notes = r"""Each column is one of the three palm specifications in the paper's main table. The first row of each panel reports the coefficient on palm exposure $\times$ IFLS5 $\times T$, where $T$ is the seven-day average temperature measure in the panel, in $^\circ$C. The coefficient does not depend on the clustering. The following rows report its standard error under three choices: clustering by district of interview, as in the paper; two-way clustering by district and calendar year-month of interview; and clustering by the respondent's IFLS4 province. Stars beside each standard error indicate the significance of the coefficient under that choice: $^*p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$ (unadjusted tests). For the two-way case in columns (1) and (3) the estimated covariance matrix is not positive semi-definite, although the variance of the reported coefficient is positive."""
out["r3_10_clustering"] = doc("R3.10", "Palm Design: Alternative Clustering", body, notes)
# ---------- R2 minor 1
i = pd.read_csv(f"{B}/r2_minor1_current/analysis/results/table_i_results.csv").set_index("model_id")
c = pd.read_csv(f"{B}/r2_minor1_current/analysis/results/table_c_results.csv").set_index("model_id")
def cap(k):
    xs = [i.loc[f"table_i_palm_{k}"], i.loc[f"table_i_jobloss_{k}"], c.loc[f"table_c_transport_post_subsidy_{k}"]]
    return " & ".join(f(x.estimate,2)+st(x.p) for x in xs), " & ".join(f"({x.se:.2f})" for x in xs)
a, b = cap("upper95"), cap("upper99")
body = rf"""\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lccc@{{}}}}
\toprule
 & \multicolumn{{2}}{{c}}{{Household work income}} & \shortstack{{Transport and\\fuel spending}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-4}}
 & \shortstack{{Palm shock\\(1)}} & \shortstack{{Job loss\\(2)}} & \shortstack{{Post subsidy cut\\(3)}} \\
\midrule
\multicolumn{{4}}{{l}}{{\textit{{Panel A. Winsorized at the 95th percentile (paper)}}}} \\
\addlinespace[4pt]
Shock effect & {a[0]} \\
 & {a[1]} \\
\addlinespace[8pt]
\multicolumn{{4}}{{l}}{{\textit{{Panel B. Winsorized at the 99th percentile}}}} \\
\addlinespace[4pt]
Shock effect & {b[0]} \\
 & {b[1]} \\
\addlinespace[8pt]
\midrule
Kecamatan FE & Yes & Yes & Yes \\
Month-year FE & Yes & Yes & Yes \\
Demographic controls & Yes & Yes & No \\
\midrule
Observations & 44,383 & 50,418 & 7,101 \\
District clusters & 277 & 287 & 197 \\
\bottomrule
\end{{tabular*}}"""
notes = r"""Outcomes are monthly amounts in constant 2010 USD. Columns (1) and (2) reproduce the paper's table on the direct economic effect of the stressors: the coefficient is palm-proxy household $\times$ IFLS5 in column (1) and recent job loss in column (2). Column (3) reproduces the balance-table regression cited in the text: household transport and fuel spending regressed on the post-cut indicator among IFLS5 respondents in urban vehicle-owning households. Within each column the two panels use the same observations and differ only in the percentile at which the outcome is capped. The first percentile of each outcome is zero, so capping the upper tail is equivalent to winsorizing both tails. """ + STD
out["r2_minor1_winsorizing"] = doc("R2.M1", "Economic Outcomes: Alternative Winsorizing", body, notes)
# ---------- R2 minor 2
body = r"""\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
 & \multicolumn{4}{c}{Dependent variable: CES-D z-score} \\
\cmidrule(lr){2-5}
 & \multicolumn{2}{c}{Mean temperature} & \multicolumn{2}{c}{Wet-bulb temperature} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
 & \shortstack{Linear\\(1)} & \shortstack{Quadratic\\(2)} & \shortstack{Linear\\(3)} & \shortstack{Quadratic\\(4)} \\
\midrule
$T$ & 0.001 & 0.001 & $-$0.026* & $-$0.025 \\
 & (0.012) & (0.013) & (0.015) & (0.018) \\
\addlinespace[4pt]
$T^2$ &  & 0.000 &  & 0.001 \\
 &  & (0.003) &  & (0.004) \\
\addlinespace[6pt]
\midrule
$p$-value: $T^2=0$ &  & 0.959 &  & 0.862 \\
$p$-value: all temperature terms $=0$ & 0.910 & 0.992 & 0.081 & 0.207 \\
\addlinespace[4pt]
Kecamatan FE & Yes & Yes & Yes & Yes \\
Month-year FE & Yes & Yes & Yes & Yes \\
Demographic controls & Yes & Yes & Yes & Yes \\
\midrule
Observations & 58,906 & 58,906 & 58,906 & 58,906 \\
Respondents & 38,703 & 38,703 & 38,703 & 38,703 \\
District clusters & 289 & 289 & 289 & 289 \\
\bottomrule
\end{tabular*}"""
notes = r"""Columns (1) and (3) reproduce the paper's estimates of the average effect of heat. Columns (2) and (4) add the square of temperature. $T$ is the seven-day average temperature measure, in $^\circ$C. In the quadratic columns $T$ is centered at the sample median (25.3$^\circ$C for mean temperature and 23.4$^\circ$C for wet-bulb temperature), so the coefficient on $T$ is the slope at the median. All models include the demographic controls (age, sex, education, marital status, ethnicity and religion). """ + STD
out["r2_minor2_quadratic"] = doc("R2.M2", "Average Effect of Heat: Linear and Quadratic Specifications", body, notes)
for k, v in out.items():
    open(f"{O}/{k}.tex", "w").write(v)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{k}.tex"], cwd=O, stdout=subprocess.DEVNULL)
    for e in (".aux", ".log"):
        if os.path.exists(f"{O}/{k}{e}"): os.remove(f"{O}/{k}{e}")
    subprocess.run(["pdftoppm", "-png", "-r", "60", f"{k}.pdf", f"/tmp/pv_{k}"], cwd=O)
print(sorted(os.listdir(O)))
