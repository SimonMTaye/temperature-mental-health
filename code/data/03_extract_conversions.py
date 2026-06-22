"""Extract monthly USD-per-IDR conversion factors from the IMF CSV."""

import pandas as pd

from data._schemas import CURRENCY_CONVERSIONS_SCHEMA
from data.config import GENERATED_DATA, RAW_ROOT


def main() -> None:
    (input_path,) = RAW_ROOT.glob("dataset_2026*.csv")
    data = pd.read_csv(input_path)
    keep = (
        (data["INDICATOR"] == "US Dollar per domestic currency")
        & (data["FREQUENCY"] == "Monthly")
        & (data["TYPE_OF_TRANSFORMATION"] == "Period average")
    )
    output = data.loc[keep, ["TIME_PERIOD", "OBS_VALUE"]].copy()
    output[["year", "month"]] = output["TIME_PERIOD"].str.extract(
        r"^(\d{4})-M(\d{2})$"
    ).astype(int)
    output = output.rename(columns={"OBS_VALUE": "conversion_factor"})[
        ["year", "month", "conversion_factor"]
    ].sort_values(["year", "month"])

    assert not output.empty
    output = CURRENCY_CONVERSIONS_SCHEMA.validate(output)
    GENERATED_DATA.mkdir(parents=True, exist_ok=True)
    output.to_parquet(GENERATED_DATA / "03_currency_conversions.parquet", index=False)


if __name__ == "__main__":
    main()
