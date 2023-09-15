from datetime import date
from pathlib import Path

import polars as pl

import ar_common


def is_date_supported(year: int, month: int) -> bool:
    # notebook type (1953/3~1964/3)
    start_date = date(1953, 3, 1)
    end_date = date(1964, 3, 1)
    return start_date <= date(year, month, 1) <= end_date


def scan_csv(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        dtypes={
            "no": pl.UInt16,
            "ns": pl.Categorical,
            "lat": pl.Utf8,
        },
    )


def calc_obs_date(df: pl.LazyFrame, year: int, month: int) -> pl.LazyFrame:
    # 初観測日を計算
    # 日付は存在せず、年と月のみである
    return df.with_columns(pl.date(year, month, 1).alias("first"))


def fill_blanks(df: pl.LazyFrame) -> pl.LazyFrame:
    cols: list[tuple[str, pl.PolarsDataType]] = [
        ("lat_left_sign", pl.Categorical),
        ("lat_right_sign", pl.Categorical),
        ("lat_question", pl.Categorical),
        ("lon_left", pl.UInt16),
        ("lon_right", pl.UInt16),
        ("lon_left_sign", pl.Categorical),
        ("lon_right_sign", pl.Categorical),
        ("lon_question", pl.Categorical),
        ("last", pl.Date),
        ("over", pl.Boolean),
    ]
    return df.with_columns(
        [pl.lit(None).cast(dtype).alias(col) for col, dtype in cols],
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
    df = ar_common.extract_coords_qm(df, ["lat"])
    df = ar_common.extract_coords_lr(df, ["lat"])
    df = ar_common.extract_coords_sign(df, ["lat"])
    df = ar_common.convert_lat(df)
    return fill_blanks(df)
