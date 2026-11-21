"""Table R2.4: are the palm, fuel and job-loss heat interactions equal?

Response to Reviewer 2, comment 4. Panel A re-estimates Table B columns 1, 4 and 7
separately. Panel B stacks the three specifications: each equation keeps its own
sample, regressors and fixed effects, and standard errors are clustered by district
across the stack, so respondents who appear in more than one design are accounted
for. Two Wald tests on the three interaction coefficients: all equal, and all zero.

Writes table_r2_4.tex next to this script; the file compiles on its own.

Run from the repository root:
uv run python referee_response/other_tables/R2_4_equality_test/shock_equality_table.py
"""

import pathlib

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUT = ROOT / "data" / "generated" / "30_analysis_table_input.parquet"

CONTROLS = [
    "age", "female", "edu_yrs_nullable", "married", "widowed", "divorced",
    "ethnicity_nullable", "religion_nullable",
]
HEATS = {"tmean_7d": "Mean temperature", "wetbulb_7d": "Wet-bulb temperature"}
# name: (label, group, post, IFLS5 only)
DESIGNS = {
    "palm": ("Palm shock (Table 3, column 1)", "palm_farmer_hh_ifls4", "ifls5", False),
    "fuel": ("Fuel subsidy cut (Table 3, column 4)", "urban_vehicle_hh_ifls4", "post_subsidy", True),
    "jobloss": ("Job loss (Table 3, column 7)", "job_loss_180d", None, False),
}

data = pd.read_parquet(INPUT)
data["ifls5"] = (data["wave"] == "IFLS5").astype(int)
data["month_year"] = data["month"].astype(str) + "_" + data["year"].astype(str)


def design_frame(design: str, heat: str) -> tuple[pd.DataFrame, list[str], str]:
    """Rows and regressors of one design. Returns the frame, regressor names, target term."""
    _, group, post, ifls5_only = DESIGNS[design]
    needed = ["cesd_z", heat, group, "month", "year", "gadm_fullcode", "kabupaten_full_code", *CONTROLS]
    df = data.loc[data["ifls5"] == 1] if ifls5_only else data
    df = df.dropna(subset=needed + ([post] if post else [])).copy()
    while True:
        keep = (df.groupby("gadm_fullcode")["gadm_fullcode"].transform("size") > 1) & (
            df.groupby("month_year")["month_year"].transform("size") > 1
        )
        if keep.all():
            break
        df = df.loc[keep]
    df = df.copy()
    g = df[group].astype(float)
    df["T"] = df[heat].astype(float)
    df["G"] = g
    df["G_T"] = g * df["T"]
    regressors = ["T", "G", "G_T"]
    target = "G_T"
    if post:
        p = df[post].astype(float)
        df["P"], df["G_P"], df["P_T"], df["G_P_T"] = p, g * p, p * df["T"], g * p * df["T"]
        regressors += ["P", "G_P", "P_T", "G_P_T"]
        target = "G_P_T"
    return df, regressors + CONTROLS, target


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


results = {}
for heat in HEATS:
    separate, blocks, stack_terms, targets = {}, [], [], []
    for design in DESIGNS:
        df, regressors, target = design_frame(design, heat)
        model = pf.feols(
            f"cesd_z ~ {' + '.join(regressors)} | month_year + gadm_fullcode",
            data=df, vcov={"CRV1": "kabupaten_full_code"},
        )
        row = model.tidy().loc[target]
        separate[design] = (row["Estimate"], row["Std. Error"], row["Pr(>|t|)"], int(model._N))
        # Equation-specific copies of the regressors and fixed effects for the stack.
        block = df[["cesd_z", "kabupaten_full_code", "pidlink", "wave"]].copy()
        for regressor in regressors:
            block[f"{design}__{regressor}"] = df[regressor].astype(float)
            stack_terms.append(f"{design}__{regressor}")
        block["fe_time"] = design + "_" + df["month_year"]
        block["fe_place"] = design + "_" + df["gadm_fullcode"].astype(str)
        blocks.append(block)
        targets.append(f"{design}__{target}")
    stack = pd.concat(blocks, ignore_index=True).fillna({term: 0.0 for term in stack_terms})
    stacked = pf.feols(
        f"cesd_z ~ {' + '.join(stack_terms)} | fe_time + fe_place",
        data=stack, vcov={"CRV1": "kabupaten_full_code"},
    )
    names = list(stacked.tidy().index)
    index = [names.index(term) for term in targets]
    beta = stacked.tidy()["Estimate"].values[index]
    vcov = np.asarray(stacked._vcov)[np.ix_(index, index)]
    restrictions = np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]])
    difference = restrictions @ beta
    equal = float(difference @ np.linalg.inv(restrictions @ vcov @ restrictions.T) @ difference)
    zero = float(beta @ np.linalg.inv(vcov) @ beta)
    results[heat] = {
        "separate": separate,
        "equal": (equal, stats.chi2.sf(equal, 2)),
        "zero": (zero, stats.chi2.sf(zero, 3)),
        "person_waves": len(stack.drop_duplicates(["pidlink", "wave"])),
        "clusters": stack["kabupaten_full_code"].nunique(),
    }
    print(heat, {d: tuple(round(v, 4) for v in s[:3]) + (s[3],) for d, s in separate.items()})
    print("  stack coefficients", np.round(beta, 4), "equal", results[heat]["equal"], "zero", results[heat]["zero"])


def p_text(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


heats = list(HEATS)
body = [r"\multicolumn{3}{l}{\textit{Panel A. Differential impact of heat, by design}} \\", r"\addlinespace[4pt]"]
for design, (label, *_rest) in DESIGNS.items():
    cells = [results[h]["separate"][design] for h in heats]
    body.append(f"{label} & " + " & ".join(f"{c[0]:.3f}{stars(c[2])}" for c in cells) + r" \\")
    body.append(" & " + " & ".join(f"({c[1]:.3f})" for c in cells) + r" \\")
body += [
    r"\addlinespace[8pt]",
    r"\multicolumn{3}{l}{\textit{Panel B. Tests on the three coefficients}} \\",
    r"\addlinespace[4pt]",
    r"Equal to each other: Wald $\chi^2(2)$ & " + " & ".join(f"{results[h]['equal'][0]:.2f}" for h in heats) + r" \\",
    r"\quad $p$-value & " + " & ".join(p_text(results[h]["equal"][1]) for h in heats) + r" \\",
    r"\addlinespace[4pt]",
    r"All equal to zero: Wald $\chi^2(3)$ & " + " & ".join(f"{results[h]['zero'][0]:.2f}" for h in heats) + r" \\",
    r"\quad $p$-value & " + " & ".join(p_text(results[h]["zero"][1]) for h in heats) + r" \\",
    r"\addlinespace[8pt]",
]
first = results[heats[0]]
observations = " / ".join(f"{first['separate'][d][3]:,}" for d in DESIGNS)

TABLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=0.72in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,booktabs,caption}
\captionsetup[table]{labelfont=bf,skip=12pt}
\renewcommand{\thetable}{R2.4}
\begin{document}
\thispagestyle{empty}
\begin{table}[!htbp]\centering
\caption{Equality of the Heat Interactions across the Three Designs}
\small
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcc@{}}
\toprule
 & \multicolumn{2}{c}{Dependent variable: CES-D z-score} \\
\cmidrule(lr){2-3}
 & \shortstack{Mean temperature\\(1)} & \shortstack{Wet-bulb temperature\\(2)} \\
\midrule
BODY
\midrule
Observations: palm / fuel / job loss & \multicolumn{2}{c}{OBSERVATIONS} \\
Distinct person-wave observations & \multicolumn{2}{c}{PERSONWAVES} \\
District clusters & \multicolumn{2}{c}{CLUSTERS} \\
\bottomrule
\end{tabular*}
\vspace{9pt}

\begin{minipage}{\textwidth}\footnotesize
\textit{Notes:} Panel A reports the interaction coefficient from each design, estimated separately with the specification of the corresponding column of Table 3. Coefficients are CES-D standard deviations per $1^\circ$C of the seven-day temperature measure. District-clustered standard errors appear in parentheses. $^*p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.

\smallskip
Panel B stacks the three specifications. Each keeps its own sample, controls and fixed effects, and standard errors are clustered by district across the stack, so the tests account for respondents who appear in more than one design. The first test has the null hypothesis that the three coefficients are equal; the second that all three are zero.
\end{minipage}
\end{table}
\end{document}
"""
table = (
    TABLE.replace("BODY", "\n".join(body))
    .replace("OBSERVATIONS", observations)
    .replace("PERSONWAVES", f"{first['person_waves']:,}")
    .replace("CLUSTERS", f"{first['clusters']:,}")
)
(HERE / "table_r2_4.tex").write_text(table, encoding="utf-8")
