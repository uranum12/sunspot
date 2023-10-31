import polars as pl


def fill_date(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("date").forward_fill())


def convert_number(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.cast({"no": pl.UInt8, "num": pl.UInt16})


def convert_date(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        pl.col("date").str.split("/"),
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
                    .cast(dtype)
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
                pl.min_horizontal(f"{coord}_0", f"{coord}_1").alias(
                    f"{coord}_min",
                ),
                pl.max_horizontal(f"{coord}_0", f"{coord}_1").alias(
                    f"{coord}_max",
                ),
            )
            .drop([f"{coord}_{num}" for num in [0, 1]])
        )
    return df


def sort(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.select(
        "date",
        "no",
        "num",
        "lat_max",
        "lat_min",
        "lon_max",
        "lon_min",
    ).sort("date", "no")
