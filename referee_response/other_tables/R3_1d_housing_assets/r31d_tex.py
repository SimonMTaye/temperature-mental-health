import sys, pandas as pd
O=sys.argv[1]; r=pd.read_csv(f"{O}/results.csv")
r.loc[r["var"]=="appliances","label"]="Household appliances"
def st(p): return "***" if p<.01 else "**" if p<.05 else "*" if p<.10 else ""
def f(x,d): 
    s=f"{abs(x):.{d}f}"; return ("$-$" if round(x,d)<0 else "")+s
names={"A":"Panel A. Dwelling","B":"Panel B. Utilities","C":"Panel C. Assets"}
L=[r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc@{}}",r"\toprule",
r" & \multicolumn{3}{c}{Palm-proxy households} & \multicolumn{3}{c}{Other farming households} & Difference \\",r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
r" & IFLS4 & IFLS5 & Change & IFLS4 & IFLS5 & Change & in changes \\",r" & (1) & (2) & (3) & (4) & (5) & (6) & (7) \\",r"\midrule"]
for pn in "ABC":
    L.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{names[pn]}}}}} \\"); L.append(r"\addlinespace[2pt]")
    for x in r[r.panel==pn].itertuples():
        d=1 if x.scale==100 else 2 if x.var=="rooms" else 1
        lab=x.label+(r"$^{\dagger}$" if x.scale==1 else "")
        L.append(f"{lab} & {f(x.palm_4,d)} & {f(x.palm_5,d)} & {f(x.palm_ch,d)} & {f(x.oth_4,d)} & {f(x.oth_5,d)} & {f(x.oth_ch,d)} & {f(x.diff,d)}{st(x.p)} \\\\")
        L.append(f" & & & & & & & ({x.se:.{d}f}) \\\\")
    L.append(r"\addlinespace[4pt]")
L+=[r"\midrule",r"Households & \multicolumn{3}{c}{1,233} & \multicolumn{3}{c}{2,507} & \\",r"\bottomrule",r"\end{tabular*}"]
open(f"{O}/table_r3_1d.tex","w").write("\n".join(L)+"\n")
main=r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=0.72in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,booktabs,caption}
\captionsetup[table]{labelfont=bf,skip=10pt}
\renewcommand{\thetable}{R3.1d}
\begin{document}
\thispagestyle{empty}
\begin{table}[!htbp]\centering
\caption{Housing and Asset Indicators by Survey Wave}
\label{tab:r31d}
\footnotesize
\renewcommand{\arraystretch}{0.98}
\setlength{\tabcolsep}{3pt}
\input{table_r3_1d.tex}
\vspace{6pt}

\begin{minipage}{\textwidth}\scriptsize
\textit{Notes:} Columns (1), (2), (4) and (5) report the percentage of households with each indicator; rows marked $^{\dagger}$ report means. Columns (3) and (6) report the change between waves, in percentage points (or units for rows marked $^{\dagger}$). Column (7) is column (3) minus column (6), with standard errors clustered by baseline district in parentheses. The sample is households observed in both IFLS4 and IFLS5. Palm-proxy households are agricultural households in palm-producing provinces at baseline; the comparison group is other agricultural households. Each row uses households with non-missing data in both waves (at least 1,225 palm-proxy and 2,455 other households). Panel A is recorded by the interviewer. Finished floor is tile, stone, cement or brick. Floor area is top-coded at the 99th percentile (300 m$^2$). Assets in Panel C other than refrigerator and television are ownership by any household member. $^*p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$ (unadjusted tests).
\end{minipage}
\end{table}
\end{document}
"""
open(f"{O}/main_r3_1d.tex","w").write(main)
