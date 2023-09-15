from datetime import date
from pathlib import Path

import polars as pl

import ar_common


def is_date_supported(year: int, month: int) -> bool:
    # new type (1978/2~2016/6)
    start_date = date(1978, 2, 1)
    end_date = date(2016, 6, 1)
    return start_date <= date(year, month, 1) <= end_date


def scan_csv(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        dtypes={
            "No": pl.Utf8,
            "lat": pl.Utf8,
            "lon": pl.Utf8,
            "first": pl.Utf8,
            "last": pl.Utf8,
        },
    )


def calc_obs_date(df: pl.LazyFrame, year: int, month: int) -> pl.LazyFrame:
    return (
        # 観測日を月と日に分離
        # この時月名を英語から数字へ変換
        df.with_columns(pl.col("first", "last").str.split("."))
        .with_columns(
            pl.col("first", "last")
            .list.get(0)
            .map_dict(
                {
                    "jan": 1,
                    "feb": 2,
                    "mar": 3,
                    "apr": 4,
                    "may": 5,
                    "jun": 6,
                    "jul": 7,
                    "aug": 8,
                    "sep": 9,
                    "oct": 10,
                    "nov": 11,
                    "dec": 12,
                },
            )
            .cast(pl.UInt8)
            .suffix("_month"),
            pl.col("first", "last").list.get(1).cast(pl.UInt8).suffix("_day"),
        )
        .with_columns(
            # 12月で翌年の1月までが範囲の場合、年をひとつ加算
            [
                pl.when(
                    (month == 12)  # noqa: PLR2004
                    & (pl.col(f"{obs_time}_month") == 1),
                )
                .then(
                    pl.date(
                        year + 1,
                        pl.col(f"{obs_time}_month"),
                        pl.col(f"{obs_time}_day"),
                    ),
                )
                .otherwise(
                    pl.date(
                        year,
                        pl.col(f"{obs_time}_month"),
                        pl.col(f"{obs_time}_day"),
                    ),
                )
                .alias(obs_time)
                for obs_time in ["first", "last"]
            ],
        )
        .drop("first_month", "first_day", "last_month", "last_day")
    )


def concat(data_path: Path) -> pl.LazyFrame:
    dfl: list[pl.LazyFrame] = []
    for path in data_path.glob("*-*.csv"):
        year, month = map(int, path.stem.split("-"))
        if is_date_supported(year, month):
            df = scan_csv(path)
            df = calc_obs_date(df, year, month)
            dfl.append(df)
    df = pl.concat(dfl)
    df = ar_common.extract_no(df)
    df = ar_common.detect_coords_over(df)
    df = ar_common.extract_coords_qm(df)
    df = ar_common.extract_coords_lr(df)
    df = ar_common.extract_coords_sign(df)
    df = ar_common.convert_lat(df)
    return ar_common.convert_lon(df)
