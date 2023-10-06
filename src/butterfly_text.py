from datetime import date
from pathlib import Path
from typing import TextIO

import polars as pl
from dateutil.relativedelta import relativedelta

import butterfly_common


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


def main() -> None:
    data_file = Path("out/ar/all.parquet")
    output_path = Path("out/butter")
    output_path.mkdir(parents=True, exist_ok=True)

    df_file = pl.scan_parquet(data_file).with_columns(
        # 符号あり整数値へ変換
        pl.col("lat_left", "lat_right").cast(pl.Int8),
    )

    df_file = butterfly_common.reverse_south(df_file)
    df_file = butterfly_common.reverse_minus(df_file)
    df_file = butterfly_common.fix_order(df_file)
    df_file = butterfly_common.extract_date(df_file)

    start, end = butterfly_common.calc_start_end(df_file, replace=True)

    with (output_path / "butter.txt").open("w") as file:
        # ヘッダの情報の書き込み
        write_header(file, start, end)

        current = start
        while current <= end:
            df = butterfly_common.filter_data(df_file, current).collect()

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
