from datetime import date

import polars as pl


def cast_lat_sign(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        # 符号あり整数値へ変換
        pl.col("lat_left", "lat_right").cast(pl.Int8),
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
            .alias("lat_left"),
            pl.when(pl.col("lat_left") > pl.col("lat_right"))
            .then(pl.col("lat_left"))
            .otherwise(pl.col("lat_right"))
            .alias("lat_right"),
        ],
    )


def extract_date(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        # 観測日から年と月を抽出
        *[pl.col(col).dt.year().suffix("_year") for col in ["first", "last"]],
        *[
            pl.col(col).dt.month().suffix("_month")
            for col in ["first", "last"]
        ],
    )


def filter_data_daily(df: pl.LazyFrame, date: date) -> pl.LazyFrame:
    return (
        df.filter(
            # 一日ごと区切る
            # 手帳形式では日付が月までしか存在しない
            pl.when(pl.col("last").is_null())
            .then(pl.col("first").eq(date.replace(day=1)))
            .otherwise(
                pl.lit(date).is_between(
                    pl.col("first"),
                    pl.col("last"),
                ),
            ),
        )
        .select("lat_left", "lat_right")
        .drop_nulls()
    )


def filter_data_monthly(df: pl.LazyFrame, date: date) -> pl.LazyFrame:
    return (
        df.filter(
            # 一月ごとに区切る
            # この時、first, last どちらかでも範囲に入っていれば対象とする
            (
                pl.col("first_year").eq(date.year)
                & pl.col("first_month").eq(date.month)
            )
            | (
                pl.col("last_year").eq(date.year)
                & pl.col("last_month").eq(date.month)
            ),
        )
        # 必要なデータを選択
        .select("lat_left", "lat_right")
        # null値を削除 for 1959/11 2059
        .drop_nulls()
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
