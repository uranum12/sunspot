from datetime import date

import polars as pl


def is_date_supported(year: int, month: int) -> bool:
    # ut (1960/1~1960/3) or (1968/7~2020/5)
    supported_ranges = [
        (date(1960, 1, 1), date(1960, 3, 1)),
        (date(1968, 7, 1), date(2020, 5, 1)),
    ]
    for start_date, end_date in supported_ranges:
        if start_date <= date(year, month, 1) <= end_date:
            return True
    return False


def calc_time(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.with_columns(
            # 時刻がマイナスになっているか
            pl.col("time").str.contains("-|m").alias("time_minus"),
            # 時刻から数値を取り出し、整数へ
            pl.col("time").str.extract_all(r"\d+").cast(pl.List(pl.UInt8)),
        )
        .with_columns(
            pl.when(pl.col("time_minus"))
            .then(
                pl.when(pl.col("time").list.lengths().eq(1))
                .then(
                    # 「-分」の場合
                    pl.time(
                        8,
                        60 - pl.col("time").list.get(0),
                    ),
                )
                .otherwise(
                    # 「-時:分」の場合
                    pl.time(
                        8 - pl.col("time").list.get(0),
                        60 - pl.col("time").list.get(1),
                    ),
                ),
            )
            .otherwise(
                pl.time(
                    # jst(+9:00)
                    pl.col("time").list.get(0) + 9,
                    pl.col("time").list.get(1),
                ),
            )
            # 文字列へ変換
            .dt.to_string("%H:%M")
            .alias("time"),
        )
        .drop("time_minus")
    )
