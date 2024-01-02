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


def convert_coord(
    df: pl.LazyFrame,
    *,
    col: str,
    dtype: pl.PolarsDataType,
) -> pl.LazyFrame:
    pat_left_sign = r"(?P<left_sign>[nsewpm+-]?)"
    pat_left = r"(?P<left>\d{1,2}(?:\.\d+)?)"
    pat_left = f"{pat_left_sign}{pat_left}"
    pat_right_sign = r"(?P<right_sign>[nsewpm+-]?)"
    pat_right = r"(?P<right>\d{1,2}(?:\.\d+)?)"
    pat_right = f"{pat_right_sign}{pat_right}"
    pat = f"(?i)(?:ND|{pat_left}(?:~{pat_right})?)"
    return (
        df.with_columns(
            # 正規表現で構造体へ分解
            pl.col(col).str.extract_groups(pat),
        )
        .unnest(col)  # 構造体から列へ分解
        .with_columns(
            # 符号の文字を小文字へ変換
            pl.col("left_sign", "right_sign").str.to_lowercase(),
        )
        .with_columns(
            # 右の符号と数値が存在しなければ左で埋める
            pl.col("right_sign").fill_null(pl.col("left_sign")),
            pl.col("right").fill_null(pl.col("left")),
        )
        .with_columns(
            # 右の符号が存在せず、左の符号が東西南北のマイナスの場合
            # 右の符号を左の符号で埋める
            pl.when(
                pl.col("right_sign").eq("")
                & pl.col("left_sign").is_in({"s", "w"}),
            )
            .then(pl.lit("-"))
            .otherwise(pl.col("right_sign"))
            .alias("right_sign"),
        )
        .with_columns(
            # 文字の符号を数式の符号へ変換
            pl.col("left_sign", "right_sign").str.replace_many(
                ["n", "s", "e", "w", "p", "m"],
                ["+", "-", "+", "-", "+", "-"],
            ),
        )
        .with_columns(
            # 符号を数値へ反映
            pl.concat_str("left_sign", "left").alias("left"),
            pl.concat_str("right_sign", "right").alias("right"),
        )
        .with_columns(
            # 文字列から小数へ変換し、四捨五入
            pl.col("left", "right")
            .cast(pl.Float64)
            .round(),
        )
        .with_columns(
            # 最大値と最小値を算出し、整数へ変換
            pl.min_horizontal("left", "right").cast(dtype).alias(f"{col}_min"),
            pl.max_horizontal("left", "right").cast(dtype).alias(f"{col}_max"),
        )
        .drop("left_sign", "right_sign", "left", "right")
    )


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
    df_spot = convert_coord(df_spot, col="lat", dtype=pl.Int8)
    df_spot = convert_coord(df_spot, col="lon", dtype=pl.Int16)

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
