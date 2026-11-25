"""Write every displayed table statistic to its own single-number .tex file.

Each value lands in
    output/stats/table_<name>/table_<name>_<stat>_c<column>_r<row>.tex
so the paper can quote it with \\input{...}. Columns are the table's data
columns (1) .. (k). Rows count the table's numeric rows top to bottom:
coefficient rows first, then bottom-of-table rows (Observations, R^2, group
shares, counts, ...) continue the count. Standard-error lines and
fixed-effect indicator rows (x / -) are not counted.

Stat names: coeff / se / pval for regression cells, mean / sd for summary
cells, and stat for a bottom-of-table number.

Integers (observations, counts) are written in full without separators;
other values are rounded to STATS_DECIMALS (library/config.py). The paper
formats and rounds each number with \\inputstat. Each file ends in "%" so
that a plain \\input does not add a trailing space after the number.
"""

import math
import numbers

from library.config import STATS_DECIMALS, STATS_OUTPUT

# One table cell: statistic name -> value, e.g. {"coeff": b, "se": se, "pval": p}.
CellStats = dict[str, float]
# Rows of cells, in display order.
StatsGrid = list[list[CellStats]]


def regression_cell(
    coefficient: float, standard_error: float, p_value: float
) -> CellStats:
    return {"coeff": coefficient, "se": standard_error, "pval": p_value}


def bottom_row(values: list) -> list[CellStats]:
    """One bottom-of-table row; blanks and non-numeric cells are skipped."""
    return [bottom_cell(value) for value in values]


def bottom_cell(value) -> CellStats:
    if isinstance(value, str):
        value = parse_number(value)
    if value is None:
        return {}
    return {"stat": value}


def parse_number(text: str) -> float | int | None:
    text = text.replace(",", "").strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return None


def format_stat(value: float, decimals: int = STATS_DECIMALS) -> str:
    if isinstance(value, numbers.Integral):
        return f"{value}%"
    text = f"{value:.{decimals}f}"
    if float(text) == 0:
        text = f"{0:.{decimals}f}"  # avoid "-0.00"
    return text + "%"


def write_table_stats(table_name: str, grid: StatsGrid) -> None:
    """Replace output/stats/table_<table_name>/ with one file per statistic."""
    output_dir = STATS_OUTPUT / f"table_{table_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_dir.glob(f"table_{table_name}_*.tex"):
        stale_file.unlink()

    for row_number, row in enumerate(grid, start=1):
        for column_number, cell in enumerate(row, start=1):
            for stat, value in cell.items():
                if value is None or math.isnan(value):
                    continue
                file_name = (
                    f"table_{table_name}_{stat}_c{column_number}_r{row_number}.tex"
                )
                (output_dir / file_name).write_text(
                    format_stat(value), encoding="utf-8"
                )
