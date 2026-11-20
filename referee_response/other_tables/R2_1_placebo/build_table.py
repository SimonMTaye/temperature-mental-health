"""Build Table R2.1 from results.csv in the Table B booktabs style used by the other reviewer tables."""
import pathlib, pandas as pd
H = pathlib.Path(__file__).resolve().parent
r = pd.read_csv(H / "results.csv")
def star(p): return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
def cells(heat):
    s = r[r.heat == heat].set_index("outcome").loc[["cesd_z", "function9_count_baseline_z", "grip_mean_kg_baseline_z"]]
    est = " & ".join(f"{'$-$' if e < 0 else ''}{abs(e):.3f}{star(p)}" for e, p in zip(s.estimate, s.p))
    se = " & ".join(f"({x:.3f})" for x in s.se)
    return est, se
a, b = cells("tmean_7d"), cells("wetbulb_7d")
f = r.iloc[0]
tex = rf"""\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lccc@{{}}}}
\toprule
 & \shortstack{{CES-D\\z-score\\(1)}} & \shortstack{{Physical\\limitations\\(2)}} & \shortstack{{Grip\\strength\\(3)}} \\
\midrule
\multicolumn{{4}}{{l}}{{\textit{{Panel A. Mean temperature}}}} \\
\addlinespace[4pt]
Palm proxy $\times$ IFLS5 $\times T$ & {a[0]} \\
 & {a[1]} \\
\addlinespace[8pt]
\multicolumn{{4}}{{l}}{{\textit{{Panel B. Wet-bulb temperature}}}} \\
\addlinespace[4pt]
Palm proxy $\times$ IFLS5 $\times T$ & {b[0]} \\
 & {b[1]} \\
\addlinespace[8pt]
\midrule
Individual FE & Yes & Yes & Yes \\
Kecamatan FE & Yes & Yes & Yes \\
Month-year FE & Yes & Yes & Yes \\
Demographic controls & Yes & Yes & Yes \\
\midrule
Observations & {f.n:,} & {f.n:,} & {f.n:,} \\
Respondents & {f.people:,} & {f.people:,} & {f.people:,} \\
Palm-proxy respondents & {f.palm_people:,} & {f.palm_people:,} & {f.palm_people:,} \\
District clusters & {f.clusters:,} & {f.clusters:,} & {f.clusters:,} \\
\bottomrule
\end{{tabular*}}
"""
(H / "table_r2_1.tex").write_text(tex)
