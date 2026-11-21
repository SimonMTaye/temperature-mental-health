"""Build shareable PDFs of the multiple-hypothesis-testing results.

Reads the result CSVs written by the mht_*.do files and writes one report per entry of
REPORTS to output/tables/stata_port/:

    mht_report.pdf            all corrections side by side
    mht_report_no_ri.pdf      Romano-Wolf and sharpened q-values
    mht_report_rwolf.pdf      Romano-Wolf only
    mht_report_qvalues.pdf    sharpened q-values only
    mht_report_ri.pdf         randomisation inference only

    uv run python code/analysis/stata_v2/mht_report.py
"""

import subprocess

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
# Rows each correction method adds below the clustered p: (column, label) pairs.
SINGLE_TEST_ROWS = {"ri": [("p_ri", "RI $p$")]}
CORRECTION_ROWS = {
    "rwolf": [("p_rwolf", "Romano--Wolf $p$")],
    "qvalues": [("q_sharpened", "Sharpened FDR $q$")],
    "ri": [("p_ri_fwer", "RI $p$ (family-wise)")],
}
REPORTS = {
    "mht_report": ("rwolf", "qvalues", "ri"),
    "mht_report_no_ri": ("rwolf", "qvalues"),
    "mht_report_rwolf": ("rwolf",),
    "mht_report_qvalues": ("qvalues",),
    "mht_report_ri": ("ri",),
}
METHOD_NAMES = {"rwolf": "Romano--Wolf", "qvalues": "sharpened FDR $q$-values", "ri": "randomisation inference"}


def load_results() -> pd.DataFrame:
    """One row per hypothesis with the clustered estimate and every method's adjusted p-values."""
    base = pd.read_csv(MHT_DIR / "uncorrected.csv")[["id", "b", "se", "p", "N", "N_clust"]]
    rwolf = pd.concat(pd.read_csv(MHT_DIR / f"rwolf_{f}.csv") for f in "AB")[["id", "p_rwolf", "reps"]]
    qvalues = pd.read_csv(MHT_DIR / "qvalues.csv")[["id", "q_sharpened"]]
    ri = pd.concat(pd.read_csv(MHT_DIR / f"ri_{f}.csv") for f in "AB")[["id", "p_ri", "p_ri_fwer", "perms"]]
    return base.merge(rwolf, on="id").merge(qvalues, on="id").merge(ri, on="id").set_index("id")


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


def panel_rows(results: pd.DataFrame, ids: list[str], coef_label: str, methods: tuple[str, ...]) -> list[str]:
    sub = results.loc[ids]
    rwolf_floor = 2 / (sub["reps"].iloc[0] + 1)
    lines = [
        row(coef_label, [rf"${b:.4f}{stars(p)}$" for b, p in zip(sub["b"], sub["p"])]),
        row("", [f"({se:.4f})" for se in sub["se"]]),
        r"\addlinespace[2pt]",
        r"\multicolumn{" + str(len(ids) + 1) + r"}{l}{\textit{Single test:}} \\",
        row(r"\quad Clustered $p$", [p_cell(p, 0.001) if p < 0.001 else p_cell(p) for p in sub["p"]]),
    ]
    for method in methods:
        for column, label in SINGLE_TEST_ROWS.get(method, []):
            lines.append(row(r"\quad " + label, [p_cell(p) for p in sub[column]]))
    lines += [
        r"\addlinespace[3pt]",
        r"\multicolumn{" + str(len(ids) + 1) + r"}{l}{\textit{Multiple-testing corrected:}} \\",
    ]
    for method in methods:
        for column, label in CORRECTION_ROWS[method]:
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


def family_a_table(results: pd.DataFrame, methods: tuple[str, ...]) -> str:
    ids = [f"table_a_{i}" for i in range(1, 7)]
    body = panel_rows(results, ids, "Heat", methods)
    return tabular(FAMILY_A_COLUMNS, body)


def family_b_table(results: pd.DataFrame, methods: tuple[str, ...]) -> str:
    n = len(FAMILY_B_COLUMNS)
    body = (
        [rf"\multicolumn{{{n + 1}}}{{l}}{{\textbf{{Panel A: 7-day mean temperature}}}} \\", r"\addlinespace[2pt]"]
        + panel_rows(results, [f"table_b_{i}" for i in range(1, 8)], "Differential impact of heat", methods)
        + [r"\midrule", rf"\multicolumn{{{n + 1}}}{{l}}{{\textbf{{Panel B: 7-day mean wet-bulb temperature}}}} \\", r"\addlinespace[2pt]"]
        + panel_rows(results, [f"table_b_{i}" for i in range(8, 15)], "Differential impact of heat", methods)
    )
    return tabular(FAMILY_B_COLUMNS, body)


def notes(results: pd.DataFrame, family: str, methods: tuple[str, ...]) -> str:
    rows = results.loc[results.index.str.startswith(f"table_{family.lower()}_")]
    reps = int(rows["reps"].iloc[0])
    perms = int(rows["perms"].iloc[0])
    size = 6 if family == "A" else 14
    method_notes = {
        "rwolf": rf"Romano--Wolf: step-down family-wise error rate, cluster bootstrap with {reps} replications "
        rf"(``$<${2 / (reps + 1):.3f}'' is the smallest attainable value at that number of replications). ",
        "qvalues": r"Sharpened FDR $q$: Anderson (2008) false discovery rate $q$-values. ",
        "ri": rf"RI: randomisation inference (Young's \texttt{{randcmd}}) with {perms} permutations of interview-date weather "
        r"within community $\times$ wave, using the clustered $t$-statistic; the single-test RI $p$ compares each $|t|$ with its "
        r"own permutation distribution, the family-wise one with the distribution of the largest $|t|$ in the family. ",
    }
    return (
        r"\begin{minipage}{\linewidth}\footnotesize\emph{Notes:} "
        r"Coefficients with standard errors \textbf{in parentheses}, clustered by IFLS4 (baseline) kabupaten. "
        r"All numbers \textbf{in square brackets are $p$-values}. \emph{Single test} rows test each coefficient on its own "
        r"(no multiple-testing adjustment); \emph{multiple-testing corrected} rows account for testing all "
        rf"{size} coefficients in this table as one family. "
        + "".join(method_notes[m] for m in methods)
        + r"Stars apply to the number they are attached to: * $p<0.10$, ** $p<0.05$, *** $p<0.01$."
        r"\end{minipage}"
    )


def p_value_guide(methods: tuple[str, ...]) -> str:
    single = ["clustered $p$"] + [label for m in methods for _, label in SINGLE_TEST_ROWS.get(m, [])]
    corrected = [label for m in methods for _, label in CORRECTION_ROWS[m]]
    return f"first the single-test values ({', '.join(single)}), then the multiple-testing corrected values ({', '.join(corrected)})."


def summary(results: pd.DataFrame, methods: tuple[str, ...]) -> str:
    """Romano-Wolf and q-value sentences are written by hand from the current results: update them after a rerun."""
    smallest_ri = {f: results.loc[results.index.str.startswith(f"table_{f}_"), "p_ri"].min() for f in "ab"}
    sentences = {
        "rwolf": r"No main effect of heat survives the Romano--Wolf correction. In Family B the three palm-shock interactions "
        r"survive it ($p \le 0.02$) in both panels; the fuel-subsidy and job-loss interactions do not (Romano--Wolf $p$ 0.14--0.68).",
        "qvalues": r"Under false-discovery-rate control no main effect of heat is significant (all $q = 1$), while in Family B every "
        r"interaction except the cash-transfer group is (sharpened $q \le 0.05$).",
        "ri": rf"The smallest single-test RI $p$ is {smallest_ri['a']:.3f} in Family A and {smallest_ri['b']:.3f} in Family B.",
    }
    return " ".join(sentences[m] for m in methods)


def ri_caveat(methods: tuple[str, ...]) -> str:
    if "ri" not in methods:
        return ""
    return r"""\medskip
\noindent\textbf{Caveat on randomisation inference.} The permutation reshuffles interview dates within each community and survey
wave, so it holds each community-wave's average weather fixed and only randomises the day-to-day deviations around it. The shock
$\times$ heat interactions are identified almost entirely by the community-wave component (palm: $t = 4.15$ on the community-wave
mean vs.\ $t = -0.17$ on daily deviations), so this test has little power against them by construction. RI $p$-values here should be
read as ``the effect is not driven by daily weather wiggles'', not as evidence against it.
"""


def title(methods: tuple[str, ...]) -> str:
    if len(methods) == 1:
        return "Heat and CES-D: " + METHOD_NAMES[methods[0]]
    return "Heat and CES-D: multiple-hypothesis corrections"


def document(results: pd.DataFrame, methods: tuple[str, ...]) -> str:
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=2cm]{{geometry}}
\usepackage{{booktabs,adjustbox,caption,textcomp,lmodern}}
\usepackage[T1]{{fontenc}}
\captionsetup{{labelfont=bf,font=normalsize,skip=6pt}}
\renewcommand{{\arraystretch}}{{1.15}}
\begin{{document}}

\begin{{center}}{{\Large\bfseries {title(methods)}}}\end{{center}}

\noindent Two families of hypotheses, matching the paper's tables. \textbf{{Family A}}: the main effect of heat on the CES-D
$z$-score, one coefficient for each of six heat measures (Table A). \textbf{{Family B}}: the differential impact of heat for
households hit by each economic shock (Table B), for 7-day mean temperature (Panel A) and 7-day mean wet-bulb temperature
(Panel B). Each family is corrected as a whole.

\bigskip
\noindent\textbf{{How to read the tables.}} Each column shows the coefficient (stars from the clustered $p$-value), its standard
error \textbf{{(in parentheses)}}, and then \textbf{{$p$-values [in square brackets]}}: {p_value_guide(methods)}

\bigskip
\noindent\textbf{{Summary.}} {summary(results, methods)}

{ri_caveat(methods)}
\begin{{table}}[!ht]
\centering
\caption{{Family A --- Main effect of heat on CES-D $z$-score}}
\begin{{adjustbox}}{{max width=\linewidth}}
{family_a_table(results, methods)}
\end{{adjustbox}}
\par\smallskip
{notes(results, "A", methods)}
\end{{table}}

\begin{{table}}[!ht]
\centering
\caption{{Family B --- Differential impact of heat by economic shock}}
\begin{{adjustbox}}{{max width=\linewidth}}
{family_b_table(results, methods)}
\end{{adjustbox}}
\par\smallskip
{notes(results, "B", methods)}
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
    results = load_results()
    for stem, methods in REPORTS.items():
        tex_path = OUT_DIR / f"{stem}.tex"
        tex_path.write_text(document(results, methods), encoding="utf-8")
        compile_pdf(tex_path)
        print(f"wrote {tex_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
