"""Write the single Stata port dataset: data/generated/stata_port/analysis_port.dta.

Port-only additions (not used by the Python tables):
- `kabupaten_full_code_ifls4`: the person's IFLS4 kabupaten mapped to both waves
  via pidlink; IFLS5-only entrants keep their own kabupaten. Used as the cluster
  for the multiple-hypothesis-testing stage so that an individual's two waves are
  never split across resampled clusters.
- `interview_date`: calendar date of the interview (datetime floored to the day),
  the unit of the randomisation-inference blocks (community x date).
"""
import pandas as pd

from library.port_helper import save_data
from library.specs import build_palm_df


def add_port_columns(df: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        df.loc[df["wave"] == "IFLS4", ["pidlink", "kabupaten_full_code"]]
        .drop_duplicates("pidlink")
        .rename(columns={"kabupaten_full_code": "kabupaten_full_code_ifls4"})
    )
    df = df.merge(baseline, on="pidlink", how="left", validate="m:1")
    df["kabupaten_full_code_ifls4"] = df["kabupaten_full_code_ifls4"].fillna(df["kabupaten_full_code"])
    df["interview_date"] = pd.to_datetime(df["interview_datetime"]).dt.normalize()
    return df


if __name__ == "__main__":
    save_data(add_port_columns(build_palm_df()))
