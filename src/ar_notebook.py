from pathlib import Path

import polars as pl


def scan_csv(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        dtypes={
            "no": pl.Utf8,
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


def concat_no(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        pl.col("no").str.split("_").cast(pl.List(pl.UInt32)),
    ).with_columns(
        pl.col("no").list.get(0) + pl.col("no").list.get(1),
    )
