"""Build a single TeX preview containing all final tables and figures."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
TABLE_SCRIPT_DIR = PROJECT / "code" / "analysis" / "tables"
FIGURE_SCRIPT_DIR = PROJECT / "code" / "analysis" / "figures"
TABLE_OUTPUT_DIR = PROJECT / "output" / "tables"
FIGURE_OUTPUT_DIR = PROJECT / "output" / "figures"


def discover_scripts(directory: Path, pattern: str) -> list[Path]:
    """Return final script files in stable filename order."""
    return sorted(
        path for path in directory.glob(pattern) if not path.name.startswith("_")
    )


def run_script(script: Path) -> None:
    print(f"running {script.relative_to(PROJECT)}", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=PROJECT, check=True)


def tex_title(stem: str) -> str:
    return stem.replace("_", " ").title()


def table_preview_block(table_script: Path) -> list[str]:
    table_name = table_script.stem
    table_body = TABLE_OUTPUT_DIR / f"{table_name}_body.tex"
    if not table_body.exists():
        raise FileNotFoundError(f"expected table output is missing: {table_body}")
    table_path = table_body.as_posix()
    return [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{tex_title(table_name)}}}",
        rf"\input{{{table_path}}}",
        r"\end{table}",
        "",
    ]


def figure_preview_block(figure_script: Path) -> list[str]:
    figure_name = figure_script.stem
    pdf = FIGURE_OUTPUT_DIR / f"{figure_name}.pdf"
    png = FIGURE_OUTPUT_DIR / f"{figure_name}.png"
    figure_output = pdf if pdf.exists() else png
    if not figure_output.exists():
        raise FileNotFoundError(f"expected figure output is missing: {pdf} or {png}")
    figure_path = figure_output.as_posix()
    return [
        r"\begin{figure}[H]",
        r"\centering",
        rf"\includegraphics[width=\textwidth]{{{figure_path}}}",
        rf"\caption{{{tex_title(figure_name)}}}",
        r"\end{figure}",
        "",
    ]


def preview_lines(table_scripts: list[Path], figure_scripts: list[Path]) -> list[str]:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{caption}",
        r"\usepackage{float}",
        r"\usepackage{graphicx}",
        r"\begin{document}",
        r"\section*{Tables}",
        "",
    ]
    for table_script in table_scripts:
        lines.extend(table_preview_block(table_script))

    lines.extend([r"\section*{Figures}", ""])
    for figure_script in figure_scripts:
        lines.extend(figure_preview_block(figure_script))

    lines.append(r"\end{document}")
    return lines


def render_preview(table_scripts: list[Path], figure_scripts: list[Path]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = PROJECT / f"preview_{timestamp}.pdf"

    with tempfile.TemporaryDirectory(prefix=f"preview_{timestamp}_") as build_dir_name:
        build_dir = Path(build_dir_name)
        tex_path = build_dir / f"preview_{timestamp}.tex"
        tex_path.write_text(
            "\n".join(preview_lines(table_scripts, figure_scripts)) + "\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(build_dir),
                str(tex_path),
            ],
            cwd=PROJECT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            output = "\n".join(part for part in [result.stdout, result.stderr] if part)
            raise RuntimeError(f"pdflatex failed while rendering preview:\n{output}")
        (build_dir / f"{tex_path.stem}.pdf").replace(pdf_path)

    return pdf_path


def main() -> None:
    TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table_scripts = discover_scripts(TABLE_SCRIPT_DIR, "table_*.py")
    figure_scripts = discover_scripts(FIGURE_SCRIPT_DIR, "figure_*.py")
    if not table_scripts and not figure_scripts:
        raise FileNotFoundError("no table or figure scripts found")

    for script in [*table_scripts, *figure_scripts]:
        run_script(script)

    preview_path = render_preview(table_scripts, figure_scripts)
    print(f"wrote {preview_path.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
