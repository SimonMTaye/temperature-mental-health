"""Project path configuration shared by data and analysis scripts."""
from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]

# External IFLS data roots. Raw archives live under RAW_ROOT; extracted Stata
# files live under RAW.
RAW_ROOT = Path("E:/IFLS")
RAW = RAW_ROOT / "extracted"
OUT_ROOT = RAW

# Repo-local generated artifacts.
CODE = PROJECT / "code"
DATA_CODE = CODE / "data"
ANALYSIS_CODE = CODE / "analysis"
DATA = PROJECT / "data"
GENERATED = DATA / "generated"
OUT = GENERATED
RESULTS = GENERATED / "results"
TMP_TEMPERATURE = GENERATED / "_tmp_temperature"
TMP_TEMPERATURE_HOURLY = GENERATED / "_tmp_temperature_hourly"
TMP_PM25 = GENERATED / "_tmp_pm25"

# Repo-local paper outputs.
OUTPUT = PROJECT / "output"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"

# Shared external resources.
GADM_PATH = RAW / "gadm" / "gadm41_IDN.gpkg"
GEE_ENV_PATH = Path("C:/Users/jingy/Dropbox/solar panel/.env")
HAO_IFLS = PROJECT.parent / "hao" / "IFLS"
