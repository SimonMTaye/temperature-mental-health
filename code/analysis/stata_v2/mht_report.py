"""Build a shareable PDF of the multiple-hypothesis-testing results.

Reads the result CSVs written by the mht_*.do files and writes
output/tables/stata_port/mht_report.{tex,pdf}.

    uv run python code/analysis/stata_v2/mht_report.py
    uv run python code/analysis/stata_v2/mht_report.py --debug   # + randcmd cross-check rows

--debug writes mht_report_debug.{tex,pdf} with extra rows from mht_ri_randcmd.do
(ri_randcmd_<family>.csv); columns without a randcmd run show "--".
"""

import subprocess
import sys

import pandas as pd

from library.config import PROJECT

MHT_DIR = PROJECT / ".cache" / "stata_port" / "mht"
OUT_DIR = PROJECT / "output" / "tables" / "stata_port"

FAMILY_A_COLUMNS = [
    r"\shortstack{7-day mean\\Temperature}",
    r"\shortstack{$>$30\,\textdegree C days\\in past week}",
    r"\shortstack{7-day mean\\Wet-bulb}",
    r"\shortstack{Daily mean\\Temperature}",
    r"\shortstack{Daily max\\Temperature}",
    r"\shortstack{Daily mean\\Wet-bulb}",
]
FAMILY_B_COLUMNS = [
    "Palm Shock",
    r"\shortstack{Palm Shocks\\Panel}",
    r"\shortstack{Palm Shock\\Price Drop}",
    r"\shortstack{Urban Vehicle\\Owners}",
    r"\shortstack{Urban Vehicle\\Owners: No\\Cash Transfer}",
    r"\shortstack{Urban Vehicle\\Owners: Cash\\Transfer}",
    "Job Loss",
]
SINGLE_TEST_ROWS = [
    ("p_ri", "RI $p$"),
]
CORRECTION_ROWS = [
    ("p_rwolf", "Romano--Wolf $p$"),
    ("q_sharpened", "Sharpened FDR $q$"),
    ("p_ri_fwer", "RI $p$ (family-wise)"),
]
DEBUG_SINGLE_TEST_ROWS = [("p_randcmd", r"\textcolor{red}{RI $p$, randcmd [debug]}")]
DEBUG_CORRECTION_ROWS = [("p_randcmd_fwer", r"\textcolor{red}{RI $p$ (family-wise), randcmd [debug]}")]


def load_results(debug: bool = False) -> pd.DataFrame:
    base = pd.read_csv(MHT_DIR / "uncorrected.csv")[["id", "b", "se", "p", "N", "N_clust"]]
    rwolf = pd.concat(pd.read_csv(MHT_DIR / f"rwolf_{f}.csv") for f in "AB")[["id", "p_rwolf", "reps"]]
    ri = pd.concat(pd.read_csv(MHT_DIR / f"ri_{f}.csv") for f in "AB")[["id", "p_ri", "p_ri_fwer", "perms"]]
    qvalues = pd.read_csv(MHT_DIR / "qvalues.csv")[["id", "q_sharpened"]]
    results = base.merge(rwolf, on="id").merge(ri, on="id").merge(qvalues, on="id")
    if debug:
        results = results.merge(load_randcmd(), on="id", how="left")
    return results.set_index("id")


def load_randcmd() -> pd.DataFrame:
    """randcmd cross-check results; the family-wise p is only meaningful if the whole family was run."""
    frames = []
    for family, size in (("A", 6), ("B", 14)):
        path = MHT_DIR / f"ri_randcmd_{family}.csv"
        if not path.exists():
            continue
        run = pd.read_csv(path).rename(columns={"p_ri_fwer": "p_randcmd_fwer"})
        if len(run) < size:
            run["p_randcmd_fwer"] = float("nan")
        frames.append(run[["id", "p_randcmd", "p_randcmd_fwer"]])
    if not frames:
        return pd.DataFrame(columns=["id", "p_randcmd", "p_randcmd_fwer"])
    return pd.concat(frames)


def stars(p: float) -> str:
    if p < 0.01:
        return "^{***}"
    if p < 0.05:
        return "^{**}"
    if p < 0.10:
        return "^{*}"
    return ""


def p_cell(p: float, floor: float | None = None) -> str:
    """Corrected p-value in square brackets, starred; show '<floor' at the resampling resolution limit."""
    if pd.isna(p):
        return "--"
    if floor is not None and p <= floor + 1e-12:
        return rf"$[<{floor:.3f}]{stars(p)}$"
    return rf"$[{p:.3f}]{stars(p)}$"


def row(label: str, cells: list[str]) -> str:
    return label + " & " + " & ".join(cells) + r" \\"


def panel_rows(results: pd.DataFrame, ids: list[str], coef_label: str) -> list[str]:
    sub = results.loc[ids]
    debug = "p_randcmd" in sub.columns
    rwolf_floor = 2 / (sub["reps"].iloc[0] + 1)
    lines = [
        row(coef_label, [rf"${b:.4f}{stars(p)}$" for b, p in zip(sub["b"], sub["p"])]),
        row("", [f"({se:.4f})" for se in sub["se"]]),
        r"\addlinespace[2pt]",
        r"\multicolumn{" + str(len(ids) + 1) + r"}{l}{\textit{Single test:}} \\",
        row(r"\quad Clustered $p$", [p_cell(p, 0.001) if p < 0.001 else p_cell(p) for p in sub["p"]]),
    ]
    for column, label in SINGLE_TEST_ROWS + (DEBUG_SINGLE_TEST_ROWS if debug else []):
        lines.append(row(r"\quad " + label, [p_cell(p) for p in sub[column]]))
    lines += [
        r"\addlinespace[3pt]",
        r"\multicolumn{" + str(len(ids) + 1) + r"}{l}{\textit{Multiple-testing corrected:}} \\",
    ]
    for column, label in CORRECTION_ROWS + (DEBUG_CORRECTION_ROWS if debug else []):
        floor = rwolf_floor if column == "p_rwolf" else None
        lines.append(row(r"\quad " + label, [p_cell(p, floor) for p in sub[column]]))
    lines += [
        r"\addlinespace[3pt]",
        row("Observations", [f"{int(n):,}" for n in sub["N"]]),
        row("Clusters", [f"{int(n)}" for n in sub["N_clust"]]),
    ]
    return lines


def tabular(columns: list[str], body: list[str]) -> str:
    header = row("", columns)
    numbers = row("", [f"({i})" for i in range(1, len(columns) + 1)])
    return "\n".join(
        [rf"\begin{{tabular}}{{l*{{{len(columns)}}}{{c}}}}", r"\toprule", header, numbers, r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
    )


def family_a_table(results: pd.DataFrame) -> str:
    ids = [f"table_a_{i}" for i in range(1, 7)]
    body = panel_rows(results, ids, "Heat")
    return tabular(FAMILY_A_COLUMNS, body)


def family_b_table(results: pd.DataFrame) -> str:
    n = len(FAMILY_B_COLUMNS)
    body = (
        [rf"\multicolumn{{{n + 1}}}{{l}}{{\textbf{{Panel A: 7-day mean temperature}}}} \\", r"\addlinespace[2pt]"]
        + panel_rows(results, [f"table_b_{i}" for i in range(1, 8)], "Differential impact of heat")
        + [r"\midrule", rf"\multicolumn{{{n + 1}}}{{l}}{{\textbf{{Panel B: 7-day mean wet-bulb temperature}}}} \\", r"\addlinespace[2pt]"]
        + panel_rows(results, [f"table_b_{i}" for i in range(8, 15)], "Differential impact of heat")
    )
    return tabular(FAMILY_B_COLUMNS, body)


def notes(results: pd.DataFrame, family: str) -> str:
    reps = int(results.loc[results.index.str.startswith(f"table_{family.lower()}_"), "reps"].iloc[0])
    perms = int(results.loc[results.index.str.startswith(f"table_{family.lower()}_"), "perms"].iloc[0])
    size = 6 if family == "A" else 14
    return (
        r"\begin{minipage}{\linewidth}\footnotesize\emph{Notes:} "
        r"Coefficients with standard errors \textbf{in parentheses}, clustered by IFLS4 (baseline) kabupaten. "
        r"All numbers \textbf{in square brackets are $p$-values}. \emph{Single test} rows test each coefficient on its own "
        r"(no multiple-testing adjustment); \emph{multiple-testing corrected} rows account for testing all "
        rf"{size} coefficients in this table as one family. "
        rf"Romano--Wolf: step-down family-wise error rate, cluster bootstrap with {reps} replications "
        rf"(``$<${2 / (reps + 1):.3f}'' is the smallest attainable value at that number of replications). "
        r"Sharpened FDR $q$: Anderson (2008) false discovery rate $q$-values. "
        rf"RI: randomisation inference with {perms} permutations of interview-date weather within community $\times$ wave, "
        r"using the clustered $t$-statistic; the single-test RI $p$ compares each $|t|$ with its own permutation distribution, "
        r"the family-wise one with the distribution of the largest $|t|$ in the family. "
        r"Stars apply to the number they are attached to: * $p<0.10$, ** $p<0.05$, *** $p<0.01$."
        r"\end{minipage}"
    )


def debug_note(results: pd.DataFrame) -> str:
    if "p_randcmd" not in results.columns:
        return ""
    return (
        r"\par\medskip\noindent\textcolor{red}{\textbf{Debug rows (red).} Cross-check of the RI rows using Alwyn Young's "
        r"\texttt{randcmd} (same blocks, strata, seed and permutations; \texttt{mht\_ri\_randcmd.do}). ``--'' = not run. "
        r"The family-wise version is shown only when the whole family was run.}"
    )


def document(results: pd.DataFrame) -> str:
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=2cm]{{geometry}}
\usepackage{{booktabs,adjustbox,caption,textcomp,lmodern,xcolor}}
\usepackage[T1]{{fontenc}}
\captionsetup{{labelfont=bf,font=normalsize,skip=6pt}}
\renewcommand{{\arraystretch}}{{1.15}}
\begin{{document}}

\begin{{center}}{{\Large\bfseries Heat and CES-D: multiple-hypothesis corrections}}\end{{center}}

\noindent Two families of hypotheses, matching the paper's tables. \textbf{{Family A}}: the main effect of heat on the CES-D
$z$-score, one coefficient for each of six heat measures (Table A). \textbf{{Family B}}: the differential impact of heat for
households hit by each economic shock (Table B), for 7-day mean temperature (Panel A) and 7-day mean wet-bulb temperature
(Panel B). Each family is corrected as a whole.

\bigskip
\noindent\textbf{{How to read the tables.}} Each column shows the coefficient (stars from the clustered $p$-value), its standard
error \textbf{{(in parentheses)}}, and then \textbf{{$p$-values [in square brackets]}}: two single-test $p$-values (clustered and
randomisation inference) followed by three multiple-testing corrected $p$/$q$-values.{debug_note(results)}

\bigskip
\noindent\textbf{{Summary.}} No main effect of heat survives any correction. In Family B the three palm-shock interactions
survive the Romano--Wolf family-wise correction ($p \le 0.02$) in both panels; the fuel-subsidy and job-loss interactions do not
(Romano--Wolf $p$ 0.14--0.68) but remain significant under false-discovery-rate control (sharpened $q \le 0.05$) except the
cash-transfer group.

\medskip
\noindent\textbf{{Caveat on randomisation inference.}} The permutation reshuffles interview dates within each community and survey
wave, so it holds each community-wave's average weather fixed and only randomises the day-to-day deviations around it. The shock
$\times$ heat interactions are identified almost entirely by the community-wave component (palm: $t = 4.15$ on the community-wave
mean vs.\ $t = -0.17$ on daily deviations), so this test has little power against them by construction. RI $p$-values here should be
read as ``the effect is not driven by daily weather wiggles'', not as evidence against it.

\begin{{table}}[!ht]
\centering
\caption{{Family A --- Main effect of heat on CES-D $z$-score}}
\begin{{adjustbox}}{{max width=\linewidth}}
{family_a_table(results)}
\end{{adjustbox}}
\par\smallskip
{notes(results, "A")}
\end{{table}}

\begin{{table}}[!ht]
\centering
\caption{{Family B --- Differential impact of heat by economic shock}}
\begin{{adjustbox}}{{max width=\linewidth}}
{family_b_table(results)}
\end{{adjustbox}}
\par\smallskip
{notes(results, "B")}
\end{{table}}

\end{{document}}
"""


def compile_pdf(tex_path) -> None:
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    for suffix in (".aux", ".log"):
        tex_path.with_suffix(suffix).unlink(missing_ok=True)


def main() -> None:
    debug = "--debug" in sys.argv
    results = load_results(debug)
    tex_path = OUT_DIR / ("mht_report_debug.tex" if debug else "mht_report.tex")
    tex_path.write_text(document(results), encoding="utf-8")
    compile_pdf(tex_path)
    print(f"wrote {tex_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
