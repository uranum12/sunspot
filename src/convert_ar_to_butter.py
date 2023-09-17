from datetime import date
from pathlib import Path

import polars as pl
from dateutil.relativedelta import relativedelta

df = (
    pl.scan_parquet(Path("out/ar/all.parquet"))
    .with_columns(
        # 符号あり整数値へ変換
        pl.col("lat_left", "lat_right").cast(pl.Int8),
    )
    .with_columns(
        [
            # 南半球の数値を反転
            pl.when(pl.col("ns").eq("S"))
            .then(-pl.col(col))
            .otherwise(pl.col(col))
            .alias(col)
            for col in ["lat_left", "lat_right"]
        ],
    )
    .with_columns(
        [
            # マイナスの数値を反転
            pl.when(pl.col(f"{col}_sign").eq("-"))
            .then(-pl.col(col))
            .otherwise(pl.col(col))
            .alias(col)
            for col in ["lat_left", "lat_right"]
        ],
    )
    .with_columns(
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
    .with_columns(
        # 観測日から年と月を抽出
        *[pl.col(col).dt.year().suffix("_year") for col in ["first", "last"]],
        *[
            pl.col(col).dt.month().suffix("_month")
            for col in ["first", "last"]
        ],
    )
)

start = date(1953, 3, 1)
end = date(2016, 6, 1)

with Path("out/butter.txt").open("w") as file:
    # ヘッダの情報の書き込み
    file.write("//Data File for Butterfly Diagram\n")
    file.write(f">>{start.year}/{start.month:0{2}}-")
    file.write(f"{end.year}/{end.month:0{2}}\n\n")
    file.write("<----data---->\n")

    current = start
    while current <= end:
        year = current.year
        month = current.month
        current += relativedelta(months=1)

        data = (
            df.filter(
                # 一月ごとに区切る
                # この時、first, last どちらかでも範囲に入っていれば対象とする
                (
                    pl.col("first_year").eq(year)
                    & pl.col("first_month").eq(month)
                )
                | (
                    pl.col("last_year").eq(year)
                    & pl.col("last_month").eq(month)
                ),
            )
            # 必要なデータを選択
            .select("lat_left", "lat_right")
            # 被りを削除
            .unique()
            # null値を削除 for 1959/11 2059
            .drop_nulls()
            # ソート
            .sort("lat_left", "lat_right").collect()
        )

        if data.height != 0:  # データが存在するか
            # 重なっている範囲を結合
            merged: list[list[int]] = []
            for row in data.iter_rows(named=True):
                if len(merged) == 0:
                    merged.append([row["lat_left"], row["lat_right"]])
                    continue
                if row["lat_left"] < merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], row["lat_right"])
                else:
                    merged.append([row["lat_left"], row["lat_right"]])

            # データフレームへ変換
            df_merged = pl.LazyFrame({"merged": merged}).select(
                pl.col("merged").list.get(0).alias("min"),
                pl.col("merged").list.get(1).alias("max"),
            )

            # NとSに分離
            # この時NとSに跨っているデータを分ける
            df_merged_n = (
                df_merged.filter(
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
            df_merged_s = (
                df_merged.filter(
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

            # 文字列へ変換
            items = {
                key: df.select(
                    pl.concat_str("min", "max", separator="-").alias("merged"),
                )
                .collect()
                .get_column("merged")
                .str.concat(" ")
                .item()
                for key, df in {"N": df_merged_n, "S": df_merged_s}.items()
            }
        else:
            items = {key: "" for key in ["N", "S"]}

        # N/Sそれぞれ書き込み
        for ns, i in items.items():
            file.write(f"{year}/{month:0{2}}/{ns}:{i}\n")
