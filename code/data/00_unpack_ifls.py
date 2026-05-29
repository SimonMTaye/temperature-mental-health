"""Unpack all IFLS .zip archives from E:/IFLS/ into E:/IFLS/extracted/.

Idempotent: if a target directory already has files, the archive is skipped.
Uses stdlib zipfile only — no extra deps. Run from anywhere:

    uv run python code/data/00_unpack_ifls.py
    # or
    python code/data/00_unpack_ifls.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from config import OUT_ROOT, RAW_ROOT

# Each entry: archive path (relative to RAW_ROOT) -> extraction dir (relative to OUT_ROOT).
ARCHIVES: dict[str, str] = {
    # Wave 1 (1993)
    # "IFLS1/cf93dta.zip": "IFLS1/cf93",
    # "IFLS1/hh93dta.zip": "IFLS1/hh93",
    # "IFLS1/DRU1195.zip": "IFLS1/doc",
    # # Wave 2 (1997)
    # "IFLS2/cf97dta.zip": "IFLS2/cf97",
    # "IFLS2/hh97dta.zip": "IFLS2/hh97",
    # "IFLS2/vol1_7.zip": "IFLS2/doc",
    # # Wave 3 (2000)
    # "IFLS3/cf00_all_dta.zip": "IFLS3/cf00",
    # "IFLS3/cf00_all_doc.zip": "IFLS3/cf00_doc",
    # "IFLS3/hh00_all_dta.zip": "IFLS3/hh00",
    # "IFLS3/hh00_all_doc.zip": "IFLS3/hh00_doc",
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


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_done, n_skip, n_missing = 0, 0, 0
    for rel_archive, rel_target in ARCHIVES.items():
        archive = RAW_ROOT / rel_archive
        target = OUT_ROOT / rel_target
        if not archive.exists():
            print(f"MISSING    {archive}")
            n_missing += 1
            continue
        status = unpack_one(archive, target)
        print(f"{status:10s} {archive.name:55s} -> {target}")
        if status == "extracted":
            n_done += 1
        else:
            n_skip += 1
    print(f"\nDone. extracted={n_done}  skipped={n_skip}  missing={n_missing}")


if __name__ == "__main__":
    main()
