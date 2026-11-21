"""Table R3.1e: palm panel specification with heat slopes by baseline age group.

Response to Reviewer 3, comment 1 (panel ageing). Column 1 is Table B column 2
(palm proxy, individual fixed effects). Column 2 adds heat slopes for respondents
aged 40-59 and 60+ at the IFLS4 interview (under 40 is the reference) and lets
those slopes change between waves. Panel A uses mean temperature, Panel B wet bulb.

Writes table_r3_1e.tex next to this script; the file compiles on its own.

Run from the repository root:
uv run python referee_response/other_tables/R3_1e_panel_ageing/palm_panel_ageing_table.py
"""

import pathlib

import pandas as pd
import pyfixest as pf

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUT = ROOT / "data" / "generated" / "30_analysis_table_input.parquet"

GROUP = "palm_farmer_hh_ifls4"
HEATS = {"tmean_7d": "Panel A. Mean temperature", "wetbulb_7d": "Panel B. Wet-bulb temperature"}
CONTROLS = (
    "age + female + edu_yrs_nullable + married + widowed + divorced"
    " + ethnicity_nullable + religion_nullable"
)
AGE_GROUPS = {"age_40_59": "Age 40--59", "age_60_plus": "Age 60+"}

# Sample: Table B column 2. Complete outcome, heat, palm status and controls,
# respondents observed in both waves, no singleton kecamatan or month-year cells.
df = pd.read_parquet(INPUT)
df["ifls5"] = (df["wave"] == "IFLS5").astype(int)
df = df.dropna(
    subset=[
        "cesd_z", "tmean_7d", GROUP, "month", "year", "gadm_fullcode",
        "kabupaten_full_code", *[column.strip() for column in CONTROLS.split("+")],
    ]
)
while True:
    keep = (
        (df.groupby("pidlink")["pidlink"].transform("size") > 1)
        & (df.groupby("gadm_fullcode")["gadm_fullcode"].transform("size") > 1)
        & (df.groupby(["month", "year"])["pidlink"].transform("size") > 1)
    )
    if keep.all():
        break
    df = df.loc[keep]
df = df.copy()
df["P"] = df[GROUP].astype(int)
baseline_age = df.loc[df["ifls5"] == 0].set_index("pidlink")["age"]
df["age_ifls4"] = df["pidlink"].map(baseline_age)
df["age_40_59"] = ((df["age_ifls4"] >= 40) & (df["age_ifls4"] < 60)).astype(int)
df["age_60_plus"] = (df["age_ifls4"] >= 60).astype(int)


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cell(tidy: pd.DataFrame, term: str) -> tuple[str, str]:
    if term not in tidy.index:
        return "", ""
    row = tidy.loc[term]
    estimate = f"{row['Estimate']:.3f}".replace("-", "$-$")
    return f"{estimate}{stars(row['Pr(>|t|)'])}", f"({row['Std. Error']:.3f})"


body = []
for heat, panel in HEATS.items():
    base = f"cesd_z ~ {heat} + P:ifls5:{heat} + P:{heat} + ifls5:{heat} + P:ifls5 + {CONTROLS}"
    age_terms = "".join(f" + {g}:{heat} + {g}:ifls5 + {g}:ifls5:{heat}" for g in AGE_GROUPS)
    models = [
        pf.feols(
            formula + " | month^year + gadm_fullcode + pidlink",
            data=df,
            vcov={"CRV1": "kabupaten_full_code"},
        )
        for formula in [base, base + age_terms]
    ]
    tidies = [model.tidy() for model in models]
    body += [rf"\multicolumn{{3}}{{l}}{{\textit{{{panel}}}}} \\", r"\addlinespace[4pt]"]
    rows = {f"P:ifls5:{heat}": "Palm proxy"} | {f"{g}:ifls5:{heat}": label for g, label in AGE_GROUPS.items()}
    for term, label in rows.items():
        cells = [cell(tidy, term) for tidy in tidies]
        body.append(rf"{label} $\times$ IFLS5 $\times T$ & {cells[0][0]} & {cells[1][0]} \\")
        body.append(rf" & {cells[0][1]} & {cells[1][1]} \\")
        print(heat, term, [c for c in cells])
    body.append(r"\addlinespace[8pt]")

observations = int(models[0]._N)
respondents = df["pidlink"].nunique()
palm_respondents = df.loc[df["P"] == 1, "pidlink"].nunique()
districts = df["kabupaten_full_code"].nunique()
print(observations, respondents, palm_respondents, districts)

TABLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=0.72in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,booktabs,caption}
\captionsetup[table]{labelfont=bf,skip=12pt}
\renewcommand{\thetable}{R3.1e}
\begin{document}
\thispagestyle{empty}
\begin{table}[!htbp]\centering
\caption{Palm Exposure: Panel Ageing}
\small
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcc@{}}
\toprule
 & \multicolumn{2}{c}{Dependent variable: CES-D z-score} \\
\cmidrule(lr){2-3}
 & \shortstack{Baseline\\(1)} & \shortstack{Age group $\times$\\IFLS5 $\times T$\\(2)} \\
\midrule
BODY
\midrule
Individual FE & Yes & Yes \\
Kecamatan FE & Yes & Yes \\
Month-year FE & Yes & Yes \\
Demographic controls & Yes & Yes \\
\midrule
Observations & OBS & OBS \\
Respondents & RESP & RESP \\
Palm-proxy respondents & PALM & PALM \\
District clusters & DIST & DIST \\
\bottomrule
\end{tabular*}
\vspace{9pt}

\begin{minipage}{\textwidth}\footnotesize
\textit{Notes:} Each column reports a separate specification on the same respondents observed in both waves. $T$ is the seven-day average temperature measure in the panel, in $^\circ$C. The palm proxy identifies baseline agricultural households in designated palm-producing provinces. Coefficients are CES-D standard deviations per $1^\circ$C.

\smallskip
Column (2) allows the heat slope to change between waves separately by age at the IFLS4 interview: under 40 (omitted group), 40--59, and 60 or older. The age-group rows show how much more the heat slope changed between waves for that group than for respondents under 40.

\smallskip
All models include the source demographic controls (age, sex, education, marital status, ethnicity and religion) and the estimable lower-order exposure, age-group, wave and heat terms. District-clustered standard errors appear in parentheses. $^*p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$ (unadjusted tests).
\end{minipage}
\end{table}
\end{document}
"""
table = (
    TABLE.replace("BODY", "\n".join(body))
    .replace("OBS", f"{observations:,}")
    .replace("RESP", f"{respondents:,}")
    .replace("PALM", f"{palm_respondents:,}")
    .replace("DIST", f"{districts:,}")
)
(HERE / "table_r3_1e.tex").write_text(table, encoding="utf-8")
