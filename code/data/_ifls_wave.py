"""Small IFLS4/IFLS5 wave metadata helpers for data scripts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class InterviewDateColumns:
    year: str
    month: str
    day: str


@dataclass(frozen=True)
class ScreeningColumns:
    province: str
    kabupaten_in_province: str
    kecamatan_in_kabupaten: str


@dataclass(frozen=True)
class WaveConfig:
    wave: str
    folder: Path
    hhid_col: str
    dates_file: str
    interview_date_cols: InterviewDateColumns
    year_transform: Callable[[Any], Any]
    screening_file: str
    screening_cols: ScreeningColumns


IFLS4 = WaveConfig(
    wave="IFLS4",
    folder=Path("IFLS4/hh07"),
    hhid_col="hhid07",
    dates_file="b3a_cov.dta",
    interview_date_cols=InterviewDateColumns(
        year="ivwyr1",
        month="ivwmth1",
        day="ivwday1",
    ),
    year_transform=lambda s: s + 2000,
    screening_file="bk_sc.dta",
    screening_cols=ScreeningColumns(
        province="sc010707",
        kabupaten_in_province="sc020707",
        kecamatan_in_kabupaten="sc030707",
    ),
)

IFLS5 = WaveConfig(
    wave="IFLS5",
    folder=Path("IFLS5/hh14"),
    hhid_col="hhid14",
    dates_file="b3a_time.dta",
    interview_date_cols=InterviewDateColumns(
        year="ivwyr",
        month="ivwmth",
        day="ivwday",
    ),
    year_transform=lambda s: s,
    screening_file="bk_sc1.dta",
    screening_cols=ScreeningColumns(
        province="sc01_14_14",
        kabupaten_in_province="sc02_14_14",
        kecamatan_in_kabupaten="sc03_14_14",
    ),
)


WAVE_CONFIGS: dict[str, WaveConfig] = {
    "IFLS4": IFLS4,
    "IFLS5": IFLS5,
}


def wave_config(wave: str) -> WaveConfig:
    return WAVE_CONFIGS[wave]


def wave_folder(raw: Path, wave: str) -> Path:
    return raw / WAVE_CONFIGS[wave].folder


def hhid_col(wave: str) -> str:
    return WAVE_CONFIGS[wave].hhid_col
