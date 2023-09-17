import polars as pl


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
                    pl.when(pl.col("time").list.get(1).eq(0))  # 「-時:00」の場合
                    .then(pl.time(9 - pl.col("time").list.get(0)))
                    .otherwise(
                        pl.time(
                            8 - pl.col("time").list.get(0),
                            60 - pl.col("time").list.get(1),
                        ),
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
