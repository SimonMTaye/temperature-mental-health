"""Run the IFLS data pipeline.

The raw IFLS unpacking step is idempotent: if a target directory already has
files, the archive is skipped. Run from anywhere:

    uv run python code/data/main.py
    # or
    python code/data/main.py
"""

import importlib
import sys
import zipfile
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR))

from config import RAW_IFLS, RAW_IFLS_EXTRACTED
from log import DEFAULT_LOG_FILE, configure_logging, log

# Each entry: archive path (relative to RAW_IFLS) -> extraction dir (relative to RAW_IFLS_EXTRACTED).
ARCHIVES: dict[str, str] = {
    # Wave 4 (2007)
    "IFLS4/cf07_all_dta.zip": "IFLS4/cf07",
    "IFLS4/cf07_all_doc.zip": "IFLS4/cf07_doc",
    "IFLS4/hh07_all_dta.zip": "IFLS4/hh07",
    "IFLS4/hh07_all_doc.zip": "IFLS4/hh07_doc",
    "IFLS4/crp_dta.zip": "IFLS4/crp",
    # Wave 5 (2014)
    "IFLS5/cf14_all_dta.zip": "IFLS5/cf14",
    "IFLS5/hh14_all_dta.zip": "IFLS5/hh14",
    "IFLS5/IFLS5_all_doc.zip": "IFLS5/doc",
    # Consumption / expenditure aggregates (separate release)
    "IFLS consumption_expenditure/IFLS-consumption-expenditure-aggregates.zip": "consumption/aggregates",
    "IFLS consumption_expenditure/pce-1993-1997_2000-2007.zip": "consumption/pce",
}


def unpack_one(archive: Path, target: Path) -> str:
    if target.exists() and any(target.iterdir()):
        return "skip"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    return "extracted"


def extract() -> None:
    """
    Unpack all IFLS .zip archives from RAW_IFLS into RAW_IFLS_EXTRACTED.
    """
    RAW_IFLS_EXTRACTED.mkdir(parents=True, exist_ok=True)
    n_done, n_skip, n_missing = 0, 0, 0
    archives = tqdm(ARCHIVES.items(), desc="IFLS archives", unit="archive")
    for rel_archive, rel_target in archives:
        archive = RAW_IFLS / rel_archive
        target = RAW_IFLS_EXTRACTED / rel_target
        archives.set_postfix_str(archive.name, refresh=False)
        if (not target.exists()) and (not archive.exists()):
            log(f"MISSING    {archive}", "WARNING")
            n_missing += 1
            continue
        status = unpack_one(archive, target)
        log(f"{status:10s} {archive.name:55s} -> {target}")
        if status == "extracted":
            n_done += 1
        else:
            n_skip += 1
    log(f"Done. extracted={n_done}  skipped={n_skip}  missing={n_missing}")


@dataclass(frozen=True)
class PipelineStep:
    module: str
    label: str


PIPELINE_LAYERS: tuple[tuple[PipelineStep, ...], ...] = (
    (
        PipelineStep("01_extract_individuals", "individuals"),
        PipelineStep("02_build_geography", "geography"),
    ),
    (
        PipelineStep("10_fetch_temperature_gee", "daily temperature"),
        PipelineStep("11_fetch_temperature_hourly_gee", "hourly temperature"),
        PipelineStep("12_fetch_merra_pm25_gee", "MERRA PM2.5"),
    ),
    (
        PipelineStep("20_build_economic_exposures", "economic exposures"),
        PipelineStep("21_build_health_exposures", "health exposures"),
        PipelineStep("22_build_person_covariates", "person covariates"),
        PipelineStep("23_build_finance_distress", "finance distress"),
        PipelineStep("24_score_cesd", "CES-D scores"),
        PipelineStep(
            "25_build_commodity_transport_exposures",
            "commodity/transport exposures",
        ),
    ),
    (
        PipelineStep("26_process_temperature_data", "processed temperature"),
        PipelineStep("27_build_income_mechanism_inputs", "income mechanisms"),
        PipelineStep("28_build_sleep_duration", "sleep duration"),
    ),
    (PipelineStep("30_build_analysis_table_input", "analysis table input"),),
)


def run_step(step: PipelineStep) -> None:
    log(f"--- start {step.module}: {step.label} ---")
    module = importlib.import_module(step.module)
    module.main()
    log(f"--- done  {step.module}: {step.label} ---")


def run_layer(layer_index: int, steps: tuple[PipelineStep, ...]) -> None:
    log(f"=== layer {layer_index}: {', '.join(step.module for step in steps)} ===")
    if len(steps) == 1:
        with tqdm(
            steps,
            desc=f"layer {layer_index}",
            unit="step",
            leave=True,
        ) as progress:
            for step in progress:
                progress.set_postfix_str(step.label, refresh=False)
                run_step(step)
        return

    with ThreadPoolExecutor(max_workers=len(steps)) as executor:
        futures = {executor.submit(run_step, step): step for step in steps}
        with tqdm(
            total=len(steps),
            desc=f"layer {layer_index}",
            unit="step",
            leave=True,
        ) as progress:
            for future in as_completed(futures):
                step = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"{step.module} failed") from exc
                progress.set_postfix_str(step.label, refresh=False)
                progress.update()


def parse_args() -> tuple[str, Path]:
    parser = ArgumentParser(description="Run the IFLS data pipeline.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Minimum level written to the pipeline log.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help="Path to the pipeline log file.",
    )
    args = parser.parse_args()
    return args.log_level, args.log_file


def main() -> None:
    """
    Run all the pipelines to build all datasets, in the correct order with parallelism for the same layer (i.e. layer 0, then 1, then 2)
    """
    log_level, log_file = parse_args()
    configure_logging(output="file", level=log_level, log_file=log_file)
    extract()
    layers = tqdm(
        enumerate(PIPELINE_LAYERS),
        total=len(PIPELINE_LAYERS),
        desc="pipeline layers",
        unit="layer",
    )
    for layer_index, steps in layers:
        layers.set_postfix_str(f"layer {layer_index}", refresh=False)
        run_layer(layer_index, steps)


if __name__ == "__main__":
    main()
