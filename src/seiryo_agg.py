from pathlib import Path

import polars as pl


def fill_date(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(pl.col("date").forward_fill())


def convert_number(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.cast({"no": pl.UInt8, "num": pl.UInt16})


def convert_date(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        pl.col("date").str.replace_all(r"([-/\.])", " ").str.split(by=" "),
    ).with_columns(
        pl.date(
            pl.col("date").list.get(0),
            pl.col("date").list.get(1),
            pl.col("date").list.get(2),
        ),
    )


def convert_lat(df: pl.LazyFrame) -> pl.LazyFrame:
    for coord in ["lat", "lon"]:
        pattern = r"([NS])" if coord == "lat" else r"([EW])"
        char_minus = "S" if coord == "lat" else "W"
        dtype = pl.Int8 if coord == "lat" else pl.Int16
        df = (
            df.with_columns(  # 小文字を大文字に変更
                pl.col(coord).str.to_uppercase(),
            )
            .with_columns(  # 左右を分離
                pl.col(coord).str.split("~"),
            )
            .with_columns(  # 別の列へ分離
                [
                    pl.col(coord).list.get(num).alias(f"{coord}_{num}")
                    for num in [0, 1]
                ],
            )
            .drop(coord)
            .with_columns(  # 右が存在しなければ左で埋める
                pl.col(f"{coord}_1").fill_null(pl.col(f"{coord}_0")),
            )
            .with_columns(  # 東西南北で表された方位を分離
                [
                    pl.col(f"{coord}_{num}")
                    .str.extract(pattern)
                    .alias(f"{coord}_{num}_sign")
                    for num in [0, 1]
                ],
            )
            .with_columns(  # 東西南北の文字を消去し整数へ
                [
                    pl.col(f"{coord}_{num}")
                    .str.replace(pattern, "")
                    .str.replace_many(["P", "M"], ["+", "-"])  # P -> +, M -> -
                    .cast(pl.Float64)
                    .round(0)  #  四捨五入
                    for num in [0, 1]
                ],
            )
            .with_columns(  # 右の文字が存在しなければ左の文字で埋める
                pl.col(f"{coord}_1_sign").fill_null(pl.col(f"{coord}_0_sign")),
            )
            .with_columns(  # 文字を符号に反映
                [
                    pl.when(pl.col(f"{coord}_{num}_sign").eq(char_minus))
                    .then(-pl.col(f"{coord}_{num}"))
                    .otherwise(pl.col(f"{coord}_{num}"))
                    for num in [0, 1]
                ],
            )
            .drop([f"{coord}_{num}_sign" for num in [0, 1]])
            .with_columns(  # 左右の大小関係を修正
                pl.min_horizontal(f"{coord}_0", f"{coord}_1")
                .cast(dtype)
                .alias(f"{coord}_min"),
                pl.max_horizontal(f"{coord}_0", f"{coord}_1")
                .cast(dtype)
                .alias(f"{coord}_max"),
            )
            .drop([f"{coord}_{num}" for num in [0, 1]])
        )
    return df


def split(df: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    df_spot = df.filter(~pl.col("no").eq(0))
    df_nospot = df.filter(pl.col("no").eq(0))

    return df_spot, df_nospot


def calc_lat(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.with_columns(  # 緯度の中央値を算出
            ((pl.col("lat_min") + pl.col("lat_max")) / 2).alias("lat"),
        )
        .with_columns(  # 緯度の中央値をもとに北半球と南半球を分類
            pl.when(pl.col("lat") >= 0)
            .then(pl.lit("N"))
            .otherwise(pl.lit("S"))
            .alias("lat"),
        )
        .drop("lat_min", "lat_max")
    )


def calc_sn(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.group_by("date")  # 日付ごとに集計
        .agg(
            # 北半球と南半球を分類
            pl.col("num").filter(pl.col("lat").eq("N")).alias("n"),
            pl.col("num").filter(pl.col("lat").eq("S")).alias("s"),
            # 黒点数、黒点群数の合計値を算出
            pl.col("num").count().cast(pl.UInt8).alias("tg"),
            pl.col("num").sum().cast(pl.UInt16).alias("tf"),
        )
        .with_columns(
            # 北半球、南半球それぞれの黒点数、黒点群数を算出
            pl.col("n").list.len().cast(pl.UInt8).alias("ng"),
            pl.col("n").list.sum().cast(pl.UInt16).alias("nf"),
            pl.col("s").list.len().cast(pl.UInt8).alias("sg"),
            pl.col("s").list.sum().cast(pl.UInt16).alias("sf"),
        )
        .drop("n", "s")
    )


def fill_sn(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        *[pl.lit(0).cast(pl.UInt8).alias(col) for col in ["ng", "sg", "tg"]],
        *[pl.lit(0).cast(pl.UInt16).alias(col) for col in ["nf", "sf", "tf"]],
    )


def sort_ar(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.select(
        ["date", "no", "lat_min", "lat_max", "lon_min", "lon_max"],
    ).sort("date", "no")


def sort_sn(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.select(
        ["date", "ng", "nf", "sg", "sf", "tg", "tf"],
    ).sort("date")


def main() -> None:
    path_seiryo = Path("data/seiryo")
    output_path = Path("out/seiryo")
    output_path.mkdir(parents=True, exist_ok=True)

    dfl: list[pl.LazyFrame] = []
    for path in path_seiryo.glob("*.csv"):
        df = pl.scan_csv(path, infer_schema_length=0).pipe(fill_date)
        dfl.append(df)
    df_all = pl.concat(dfl).pipe(convert_number).pipe(convert_date)

    df_spot, df_nospot = split(df_all)
    df_spot = df_spot.pipe(convert_lat)

    df_ar = df_spot.drop("num").pipe(sort_ar).collect()
    print("ar data:")
    print(df_ar)
    df_ar.write_parquet(output_path / "ar.parquet")

    df_spot = df_spot.pipe(calc_lat).pipe(calc_sn)
    df_nospot = df_nospot.select("date").pipe(fill_sn)
    df_sn = pl.concat([df_spot, df_nospot]).pipe(sort_sn).collect()
    print("sn data:")
    print(df_sn)
    df_sn.write_parquet(output_path / "sn.parquet")


if __name__ == "__main__":
    main()
