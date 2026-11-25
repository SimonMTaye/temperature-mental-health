from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "output"
TABLE_OUTPUT = OUTPUT / "tables"
FIGURE_OUTPUT = OUTPUT / "figures"

# Single-number .tex files (one per coefficient / SE / p-value) for \input in the paper.
STATS_OUTPUT = OUTPUT / "stats"
STATS_DECIMALS = 2
