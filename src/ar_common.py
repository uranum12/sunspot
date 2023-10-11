import polars as pl


def extract_ns(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        # NSデータ
        pl.col("no").str.extract(r"([NS])").cast(pl.Categorical).alias("ns"),
        # 通し番号からNSを削除
        pl.col("no").str.replace(r"[NS]", ""),
    )


def convert_no(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        # 通し番号を文字列から整数へ
        pl.col("no").cast(pl.UInt32),
    )


def detect_coords_over(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        # 経緯度が先月からの続きであるかどうか
        # 続きであった場合、それを示す行を追加し、元データ削除
        ((pl.col("lat") == "/") | (pl.col("lon") == "/")).alias("over"),
        pl.when(pl.col("lat") != "/").then(pl.col("lat")),
        pl.when(pl.col("lon") != "/").then(pl.col("lon")),
    )


def extract_coords_qm(
    df: pl.LazyFrame,
    coords: list[str] | None = None,
) -> pl.LazyFrame:
    if coords is None:
        coords = ["lat", "lon"]
    return df.with_columns(
        # 経緯度のはてなマークの存在
        pl.col(coords)
        .str.extract(r"(\?)")
        .cast(pl.Categorical)
        .suffix("_question"),
        # 経緯度からはてなマーク削除
        pl.col(coords).str.replace(r"\?", ""),
    )


def extract_coords_lr(
    df: pl.LazyFrame,
    coords: list[str] | None = None,
) -> pl.LazyFrame:
    if coords is None:
        coords = ["lat", "lon"]
    return (
        df.with_columns(
            # 経緯度の数値を分割
            pl.col(coords).str.split("~"),
        )
        .with_columns(
            # 経緯度の左の数値
            pl.col(coords)
            .list.get(0)
            .suffix("_left"),
        )
        .with_columns(
            # 経緯度の右の数値
            # 存在しない場合はここで左の数値を持ってくる
            [
                pl.when(pl.col(coord).list.len().eq(2))
                .then(pl.col(coord).list.get(1))
                .otherwise(pl.col(f"{coord}_left"))
                .suffix("_right")
                for coord in coords
            ],
        )
        .drop(coords)
    )


def extract_coords_sign(
    df: pl.LazyFrame,
    coords: list[str] | None = None,
) -> pl.LazyFrame:
    if coords is None:
        coords = ["lat", "lon"]
    return df.with_columns(
        # 経緯度の左右の数値の符号
        pl.col(
            [f"{coord}_{lr}" for coord in coords for lr in ["left", "right"]],
        )
        .str.extract(r"([p-])")
        .str.replace(r"(p)", "+")
        .cast(pl.Categorical)
        .suffix("_sign"),
        # 経緯度の左右の数値から符号を削除
        pl.col(
            [f"{coord}_{lr}" for coord in coords for lr in ["left", "right"]],
        ).str.replace(r"p|-", ""),
    )


def convert_lat(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        # 緯度の各数値を符号なし整数へ変換
        pl.col("lat_left", "lat_right").cast(pl.UInt8),
    )


def convert_lon(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        # 経度の各数値を符号なし整数へ変換
        pl.col("lon_left", "lon_right").cast(pl.UInt16),
    )


def sort(df: pl.LazyFrame) -> pl.LazyFrame:
    # 列の順番を揃え、N/Sと通し番号でソートする
    return df.select(
        "ns",
        "no",
        "lat_left",
        "lat_right",
        "lat_left_sign",
        "lat_right_sign",
        "lat_question",
        "lon_left",
        "lon_right",
        "lon_left_sign",
        "lon_right_sign",
        "lon_question",
        "first",
        "last",
        "over",
    ).sort("ns", "no")
