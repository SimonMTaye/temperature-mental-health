"""Extract monthly US CPI-U from FRED and compute the 2010-base deflator.

The deflator converts nominal USD (at the interview date) to constant 2010 USD:
    real_2010_usd = nominal_usd * cpi_deflator_2010base
where
    cpi_deflator_2010base = CPI_2010_avg / CPI_interview_month

Series: CPIAUCSL (Consumer Price Index for All Urban Consumers: All Items)
        Monthly, 1982-1984=100, Seasonally Adjusted
Source: Federal Reserve Economic Data (FRED), St. Louis Fed
        https://fred.stlouisfed.org/series/CPIAUCSL

Output: data/generated/04_us_cpi.parquet
"""

import json
import urllib.request

import pandas as pd

from data.config import GENERATED_DATA
from data._schemas import US_CPI_SCHEMA
from library.log import log

FRED_API_KEY = "95fc142d0cd372f5f2e5f2a22fd01323"
FRED_SERIES = "CPIAUCSL"
START_DATE = "2007-01-01"
END_DATE = "2015-12-01"
CPI_BASE_YEAR = 2010


def fetch_cpi_series() -> list[dict]:
    """Fetch CPIAUCSL observations from FRED."""
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={FRED_SERIES}"
        f"&api_key={FRED_API_KEY}"
        f"&file_type=json"
        f"&observation_start={START_DATE}"
        f"&observation_end={END_DATE}"
        f"&sort_order=asc"
    )
    log(f"Fetching {FRED_SERIES} from FRED ...")
    with urllib.request.urlopen(url) as resp:
        payload = json.loads(resp.read())
    observations = payload["observations"]
    log(f"Received {len(observations)} monthly observations")
    return observations


def build_deflator(observations: list[dict]) -> pd.DataFrame:
    """Build a monthly CPI dataframe with a 2010-base deflator column."""
    records = []
    for obs in observations:
        date_str = obs["date"]  # e.g. "2007-01-01"
        year, month, _ = date_str.split("-")
        cpi = float(obs["value"])
        records.append({"year": int(year), "month": int(month), "cpi_u": cpi})

    df = pd.DataFrame(records).sort_values(["year", "month"]).reset_index(drop=True)

    # Compute average CPI-U for the base year (2010).
    cpi_2010 = df.loc[df.year == CPI_BASE_YEAR, "cpi_u"]
    if cpi_2010.empty:
        raise RuntimeError(f"No observations found for base year {CPI_BASE_YEAR}")
    cpi_2010_avg = cpi_2010.mean()
    log(f"CPI 2010 average: {cpi_2010_avg:.3f}")

    # Deflator > 1: earlier dollars were worth more, scale UP to 2010.
    # Deflator < 1: later dollars were worth less, scale DOWN to 2010.
    df["cpi_deflator_2010base"] = cpi_2010_avg / df["cpi_u"]

    log(
        f"Deflator range: [{df.cpi_deflator_2010base.min():.3f}, "
        f"{df.cpi_deflator_2010base.max():.3f}]"
    )
    return df


def main() -> None:
    GENERATED_DATA.mkdir(parents=True, exist_ok=True)

    observations = fetch_cpi_series()
    df = build_deflator(observations)

    df = US_CPI_SCHEMA.validate(df)
    out_path = GENERATED_DATA / "04_us_cpi.parquet"
    df.to_parquet(out_path, index=False)
    log(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
