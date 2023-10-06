from datetime import date
from pathlib import Path
from typing import TextIO

import polars as pl
from dateutil.relativedelta import relativedelta


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


def filter_data(df: pl.LazyFrame, date: date) -> pl.LazyFrame:
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
        # 被りを削除
        .unique()
        # null値を削除 for 1959/11 2059
        .drop_nulls()
        # ソート
        .sort("lat_left", "lat_right")
    )


def merge_data(df: pl.DataFrame) -> pl.DataFrame:
    # 重なっている範囲を結合
    merged: list[list[int]] = []
    for row in df.iter_rows(named=True):
        if len(merged) == 0:
            merged.append([row["lat_left"], row["lat_right"]])
            continue
        if row["lat_left"] < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], row["lat_right"])
        else:
            merged.append([row["lat_left"], row["lat_right"]])

    # データフレームへ変換
    return pl.DataFrame({"merged": merged}).select(
        pl.col("merged").list.get(0).alias("min"),
        pl.col("merged").list.get(1).alias("max"),
    )


def split_data(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    # NとSに分離
    # この時NとSに跨っているデータを分ける
    df_n = (
        df.filter(
            pl.any_horizontal(pl.all().ge(0)),
        )
        .with_columns(
            [
                pl.when(pl.col(col).lt(0))
                .then(pl.lit(0))
                .otherwise(pl.col(col))
                .alias(col)
                for col in ["min", "max"]
            ],
        )
        .sort("min", "max")
    )
    df_s = (
        df.filter(
            pl.any_horizontal(pl.all().le(0)),
        )
        .with_columns(
            [
                pl.when(pl.col(col).gt(0))
                .then(pl.lit(0))
                .otherwise(-pl.col(col))  # Sのマイナス表記を戻す
                .alias(col)
                for col in ["min", "max"]
            ],
        )
        .rename({"min": "max", "max": "min"})
        .sort("min", "max")
    )
    return df_n, df_s


def convert_to_str(df: pl.DataFrame) -> str:
    return (
        df.select(pl.concat_str("min", "max", separator="-").alias("merged"))
        .get_column("merged")
        .str.concat(" ")
        .item()
    )


def write_header(file: TextIO, start: date, end: date) -> None:
    file.write("//Data File for Butterfly Diagram\n")
    file.write(f">>{start.year}/{start.month:0{2}}-")
    file.write(f"{end.year}/{end.month:0{2}}\n\n")
    file.write("<----data---->\n")


def write_data(file: TextIO, date: date, data_n: str, data_s: str) -> None:
    file.write(f"{date.year}/{date.month:0{2}}/N:{data_n}\n")
    file.write(f"{date.year}/{date.month:0{2}}/S:{data_s}\n")


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


def main() -> None:
    df_file = pl.scan_parquet(Path("out/ar/all.parquet")).with_columns(
        # 符号あり整数値へ変換
        pl.col("lat_left", "lat_right").cast(pl.Int8),
    )

    df_file = reverse_south(df_file)
    df_file = reverse_minus(df_file)
    df_file = fix_order(df_file)
    df_file = extract_date(df_file)

    start, end = calc_start_end(df_file, replace=True)

    with Path("out/butter.txt").open("w") as file:
        # ヘッダの情報の書き込み
        write_header(file, start, end)

        current = start
        while current <= end:
            df = filter_data(df_file, current).collect()

            if df.height != 0:  # データが存在するか
                df = merge_data(df)
                df_n, df_s = split_data(df)
                data_n = convert_to_str(df_n)
                data_s = convert_to_str(df_s)
                write_data(file, current, data_n, data_s)
            else:
                write_data(file, current, "", "")

            current += relativedelta(months=1)


if __name__ == "__main__":
    main()
