from pathlib import Path

import polars as pl


def scan_csv(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        dtypes={
            "date": pl.UInt8,
            "time": pl.Utf8,
            "ng": pl.UInt8,
            "nf": pl.UInt16,
            "sg": pl.UInt8,
            "sf": pl.UInt16,
            "remarks": pl.Utf8,
        },
    )


def calc_date(df: pl.LazyFrame, year: int, month: int) -> pl.LazyFrame:
    return df.with_columns(
        # 日付に年と月を付与
        pl.date(year, month, pl.col("date")).alias("date"),
    )


def sort(df: pl.LazyFrame) -> pl.LazyFrame:
    # 列の順番を揃え、日付でソートする
    return df.select(
        "date",
        "time",
        "ng",
        "nf",
        "sg",
        "sf",
        "remarks",
    ).sort("date")
