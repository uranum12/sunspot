import polars as pl


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
