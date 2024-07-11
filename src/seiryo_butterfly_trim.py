import json
from datetime import date
from pathlib import Path
from pprint import pprint

import polars as pl
from dateutil.relativedelta import relativedelta

from seiryo_butterfly import ButterflyInfo, fill_lat


def trim_info(
    info: ButterflyInfo,
    lat_min: int | None = None,
    lat_max: int | None = None,
    date_start: date | None = None,
    date_end: date | None = None,
) -> ButterflyInfo:
    return ButterflyInfo(
        lat_min if lat_min is not None else info.lat_min,
        lat_max if lat_max is not None else info.lat_max,
        date_start if date_start is not None else info.date_start,
        date_end if date_end is not None else info.date_end,
        info.date_interval,
    )


def trim_data(df: pl.DataFrame, info: ButterflyInfo) -> pl.DataFrame:
    return (
        df.lazy()
        .filter(pl.col("date").is_between(info.date_start, info.date_end))
        .pipe(
            fill_lat,
            start=info.date_start,
            end=info.date_end,
            interval=info.date_interval.to_interval(),
        )
        .collect()
    )


def main() -> None:
    monthly_data_path = Path("out/seiryo/butterfly/monthly.parquet")
    monthly_info_path = monthly_data_path.with_suffix(".json")
    fromtext_data_path = Path("out/seiryo/butterfly/fromtext.parquet")
    fromtext_info_path = fromtext_data_path.with_suffix(".json")
    output_path = Path("out/seiryo/butterfly")

    monthly_data = pl.read_parquet(monthly_data_path)

    with monthly_info_path.open("r") as f_monthly_info:
        monthly_info = ButterflyInfo.from_dict(json.load(f_monthly_info))

    fromtext_data = pl.read_parquet(fromtext_data_path)

    with fromtext_info_path.open("r") as f_fromtext_info:
        fromtext_info = ButterflyInfo.from_dict(json.load(f_fromtext_info))

    trimmed_monthly_info = trim_info(monthly_info, lat_min=-50, lat_max=50)
    pprint(trimmed_monthly_info)
    trimmed_monthly_data = trim_data(monthly_data, trimmed_monthly_info)
    print(trimmed_monthly_data)
    trimmed_monthly_data.write_parquet(output_path / "trimmed_monthly.parquet")
    with (output_path / "trimmed_monthly.json").open("w") as f_trimmed_monthly:
        f_trimmed_monthly.write(trimmed_monthly_info.to_json())

    trimmed_fromtext_info = trim_info(
        fromtext_info,
        lat_min=-50,
        lat_max=50,
        date_end=monthly_info.date_start - relativedelta(months=1),
    )
    pprint(trimmed_fromtext_info)
    trimmed_fromtext_data = trim_data(fromtext_data, trimmed_fromtext_info)
    print(trimmed_fromtext_data)
    trimmed_fromtext_data.write_parquet(
        output_path / "trimmed_fromtext.parquet"
    )
    with (output_path / "trimmed_fromtext.json").open(
        "w"
    ) as f_trimmed_fromtext:
        f_trimmed_fromtext.write(trimmed_fromtext_info.to_json())


if __name__ == "__main__":
    main()
