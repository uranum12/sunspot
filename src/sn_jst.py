from datetime import date

import polars as pl


def is_date_supported(year: int, month: int) -> bool:
    # jst (1954/1~1959/12) or (1960/4~1968/6)
    supported_ranges = [
        (date(1954, 1, 1), date(1959, 12, 1)),
        (date(1960, 4, 1), date(1968, 6, 1)),
    ]
    for start_date, end_date in supported_ranges:
        if start_date <= date(year, month, 1) <= end_date:
            return True
    return False


def calc_time(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        # 時刻を時と分に分離し、整数へ変換
        pl.col("time")
        .str.split(":")
        .cast(pl.List(pl.UInt8)),
    ).with_columns(
        # 時刻を文字列へ変換
        pl.time(
            pl.col("time").list.get(0),
            pl.col("time").list.get(1),
        )
        .dt.to_string("%H:%M")
        .alias("time"),
    )
