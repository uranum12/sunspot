import polars as pl


def calc_obs_date(df: pl.LazyFrame, year: int, month: int) -> pl.LazyFrame:
    # 初観測日を計算
    # 日付は存在せず、年と月のみである
    return df.with_columns(pl.date(year, month, 1).alias("first"))


def fill_blanks(
    df: pl.LazyFrame, cols: list[tuple[str, pl.PolarsDataType]]
) -> pl.LazyFrame:
    return df.with_columns(
        [pl.lit(None).cast(dtype).alias(col) for col, dtype in cols]
    )


def concat_no(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(pl.col("no").str.replace("_", ""))
