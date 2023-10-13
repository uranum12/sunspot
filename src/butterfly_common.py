from datetime import date

import polars as pl


def cast_lat_sign(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        # 符号あり整数値へ変換
        pl.col("lat_left", "lat_right").cast(pl.Int8),
    )


def drop_lat_null(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.filter(
        # 緯度の空白値を削除
        pl.all_horizontal(pl.col("lat_left", "lat_right").is_not_null()),
    )


def reverse_south(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        [
            # 南半球の数値を反転
            pl.when(pl.col("ns").eq("S"))
            .then(-pl.col(col))
            .otherwise(pl.col(col))
            .alias(col)
            for col in ["lat_left", "lat_right"]
        ],
    )


def reverse_minus(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        [
            # マイナスの数値を反転
            pl.when(pl.col(f"{col}_sign").eq("-"))
            .then(-pl.col(col))
            .otherwise(pl.col(col))
            .alias(col)
            for col in ["lat_left", "lat_right"]
        ],
    )


def fix_order(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        [
            # 左右の数値の大小関係が反転しているものを修正
            pl.when(pl.col("lat_left") > pl.col("lat_right"))
            .then(pl.col("lat_right"))
            .otherwise(pl.col("lat_left"))
            .alias("lat_min"),
            pl.when(pl.col("lat_left") > pl.col("lat_right"))
            .then(pl.col("lat_left"))
            .otherwise(pl.col("lat_right"))
            .alias("lat_max"),
        ],
    ).drop("lat_left", "lat_right")


def complement_last(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        # 最終観測日が空白の場合、初観測日の月の最終日を最終観測日として補完
        pl.when(pl.col("last").is_null())
        .then(pl.col("first").dt.month_end())
        .otherwise(pl.col("last"))
        .alias("last"),
    )


def truncate_day(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        # 観測日の日付を月単位で切り捨てる
        pl.col("first", "last").dt.truncate("1mo"),
    )


def filter_data(df: pl.LazyFrame, date: date) -> pl.LazyFrame:
    return df.filter(
        # データが初観測日と最終観測日の間に存在するか
        pl.lit(date).is_between(pl.col("first"), pl.col("last")),
    )


def calc_start_end(
    lf: pl.LazyFrame,
    *,
    replace: bool = False,
) -> tuple[date, date]:
    df = (
        lf.select(
            pl.col("first", "last").min().suffix("_min"),
            pl.col("first", "last").max().suffix("_max"),
        )
        .collect()
        .transpose()
    )
    start: date = df.min().item()
    end: date = df.max().item()

    if replace:
        start = start.replace(day=1)
        end = end.replace(day=1)

    return start, end
