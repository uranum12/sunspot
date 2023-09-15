from datetime import date
from pathlib import Path

import polars as pl

import ar_common


def is_date_supported(year: int, month: int) -> bool:
    # old type (1964/4~1978/1)
    start_date = date(1964, 4, 1)
    end_date = date(1978, 1, 1)
    return start_date <= date(year, month, 1) <= end_date


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


def concat(data_path: Path) -> pl.LazyFrame:
    dfl: list[pl.LazyFrame] = []
    for path in data_path.glob("*-*.csv"):
        year, month = map(int, path.stem.split("-"))
        if is_date_supported(year, month):
            df = scan_csv(path)
            df = calc_obs_date(df, year, month)
            dfl.append(df)
    df = pl.concat(dfl)
    df = ar_common.detect_coords_over(df)
    df = ar_common.extract_coords_qm(df)
    df = ar_common.extract_coords_lr(df)
    df = ar_common.extract_coords_sign(df)
    df = ar_common.convert_lat(df)
    df = ar_common.convert_lon(df)
    return ar_common.extract_no(df)
