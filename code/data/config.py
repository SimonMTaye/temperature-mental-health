"""Project path configuration shared by data and analysis scripts."""

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]

# External IFLS data roots. Raw archives live under RAW_ROOT; extracted Stata
# files live under RAW.
RAW_ROOT = PROJECT / "data" / "raw"
RAW_IFLS = RAW_ROOT / "IFLS"
RAW_IFLS_EXTRACTED = RAW_IFLS / "extracted"

# Repo-local generated artifacts.
CODE = PROJECT / "code"
DATA_CODE = CODE / "data"
ANALYSIS_CODE = CODE / "analysis"
DATA = PROJECT / "data"
GENERATED_DATA = DATA / "generated"

RESULTS = GENERATED_DATA / "results"
TMP_TEMPERATURE = GENERATED_DATA / "_tmp_temperature"
TMP_TEMPERATURE_HOURLY = GENERATED_DATA / "_tmp_temperature_hourly"
TMP_PM25 = GENERATED_DATA / "_tmp_pm25"

# Repo-local paper outputs.
OUTPUT = PROJECT / "output"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"

# Shared external resources.
GADM_PATH = RAW_ROOT / "gadm41_IDN.gpkg"

# NOTE: these uses Simon's credentials tied to the cornell account. Please ask an LLM
# on authenticating earthengine's python API for advice on setting up your own account and logging in
GEE_PROEJCT_ID = "vip-inc-vol-anal"
# GEE_ENV_PATH = Path("~/.config/earthengine/.env")
