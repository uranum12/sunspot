import polars as pl


def calc_time(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        # 時刻を時と分に分離し、整数へ変換
        pl.col("time").str.split(":").cast(pl.List(pl.UInt8))
    ).with_columns(
        pl.time(pl.col("time").list.get(0), pl.col("time").list.get(1)).alias(
            "time"
        )
    )
