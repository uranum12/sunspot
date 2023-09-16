from pathlib import Path

import polars as pl


def scan_csv(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        dtypes={
            "No": pl.Utf8,
            "lat": pl.Utf8,
            "lon": pl.Utf8,
            "first": pl.UInt8,
            "last": pl.UInt8,
        },
    )


def calc_obs_date(df: pl.LazyFrame, year: int, month: int) -> pl.LazyFrame:
    return df.with_columns(
        # 初観測日、最終観測日に年と月を付与
        [
            pl.date(year, month, pl.col(obs_time)).alias(obs_time)
            for obs_time in ["first", "last"]
        ],
    )
