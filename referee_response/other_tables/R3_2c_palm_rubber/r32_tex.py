import sys, pandas as pd, subprocess, os
O=sys.argv[1]; r=pd.read_csv(f"{O}/results.csv")
def st(p): return "***" if p<.01 else "**" if p<.05 else "*" if p<.10 else ""
def f(x): return ("$-$" if round(x,3)<0 else "")+f"{abs(x):.3f}"
cfg={"R3.2c":("r3_2c","Reported Crops and Changes in Heat Sensitivity: Palm and Rubber",r"Group $\times$ IFLS5 $\times T$",
 r"Each column is a separate regression with individual fixed effects. The coefficient shown is the baseline group named in the column interacted with IFLS5 and $T$, the seven-day mean temperature in $^\circ$C. Coefficients are CES-D standard deviations per $1^\circ$C."),
 "R3.2d":("r3_2d","Palm-Price Shock Across Reported Crop Groups: Palm and Rubber",r"Group $\times S \times T$",
 r"Each column is a separate regression with individual fixed effects. The coefficient shown is $G\times S\times T$, where $G$ is the baseline group named in the column, $S$ is the decline in the international palm-oil price between the respondent's IFLS4 and IFLS5 interview months (the paper's price exposure, switched on in IFLS5), and $T$ is the seven-day mean temperature in $^\circ$C. Coefficients are CES-D standard deviations per $1^\circ$C per unit of the price exposure.")}
for tab,(stem,title,rowlab,n1) in cfg.items():
    x=r[r.table==tab].reset_index(drop=True)
    est=" & ".join(f(e)+st(p) for e,p in zip(x.estimate,x.p)); se=" & ".join(f"({s:.3f})" for s in x.se)
    body=rf"""\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lcccc@{{}}}}
\toprule
 & \multicolumn{{4}}{{c}}{{Dependent variable: CES-D z-score}} \\
\cmidrule(lr){{2-5}}
 & \multicolumn{{2}}{{c}}{{Palm proxy}} & \multicolumn{{2}}{{c}}{{Rubber}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
 & (1) & (2) & (3) & (4) \\
\midrule
{rowlab} & {est} \\
 & {se} \\
\addlinespace[6pt]
\midrule
Reported crop $\times$ IFLS5 & No & Yes & No & Yes \\
Crop price change $\times$ IFLS5 & No & Yes & No & Yes \\
\addlinespace[4pt]
Individual FE & Yes & Yes & Yes & Yes \\
Kecamatan FE & Yes & Yes & Yes & Yes \\
Month-year FE & Yes & Yes & Yes & Yes \\
Demographic controls & Yes & Yes & Yes & Yes \\
\midrule
Observations & 40,012 & 40,012 & 40,012 & 40,012 \\
Respondents & 20,006 & 20,006 & 20,006 & 20,006 \\
District clusters & 273 & 273 & 273 & 273 \\
Exposed respondents & 2,511 & 2,511 & 687 & 687 \\
Exposed households & 1,208 & 1,208 & 328 & 328 \\
\bottomrule
\end{{tabular*}}
"""
    open(f"{O}/table_{stem}.tex","w").write(body)
    main=rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=0.72in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern,booktabs,caption}}
\captionsetup[table]{{labelfont=bf,skip=12pt}}
\renewcommand{{\thetable}}{{{tab}}}
\begin{{document}}
\thispagestyle{{empty}}
\begin{{table}}[!htbp]\centering
\caption{{{title}}}
\small
\renewcommand{{\arraystretch}}{{1.15}}
\setlength{{\tabcolsep}}{{3pt}}
\input{{table_{stem}.tex}}
\vspace{{9pt}}

\begin{{minipage}}{{\textwidth}}\footnotesize
\textit{{Notes:}} {n1}

\smallskip
The palm proxy identifies agricultural households in palm-producing provinces at IFLS4. The rubber group is households that report rubber among their two highest-value agricultural products in IFLS4. Groups are fixed at baseline and can overlap. Columns (2) and (4) add indicators for reporting rice, maize, rubber or coffee at baseline, each interacted with IFLS5, and the log change in the international price of each of these crops between the respondent's two interview months, interacted with the crop indicator and IFLS5. Respondents with missing crop reports are kept with a separate indicator.

\smallskip
All models include the demographic controls (age, sex, education, marital status, ethnicity and religion) and the lower-order group, wave and temperature terms. District-clustered standard errors appear in parentheses. $^*p<0.10$, $^{{**}}p<0.05$, $^{{***}}p<0.01$ (unadjusted tests).
\end{{minipage}}
\end{{table}}
\end{{document}}
"""
    open(f"{O}/main_{stem}.tex","w").write(main)
    subprocess.run(["pdflatex","-interaction=nonstopmode",f"main_{stem}.tex"],cwd=O,stdout=subprocess.DEVNULL)
    os.replace(f"{O}/main_{stem}.pdf",f"{O}/table_{stem}_palm_rubber.pdf")
    subprocess.run(["pdftoppm","-png","-r","80",f"table_{stem}_palm_rubber.pdf",f"prev_{stem}"],cwd=O)
