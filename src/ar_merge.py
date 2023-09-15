import polars as pl


def extract_over_no(df: pl.LazyFrame) -> pl.Series:
    # 複数のシートに分かれていて、データが二つ存在するものの番号
    return (
        df.filter(pl.col("over") & pl.col("no").count().over("no").eq(2))
        .select("no")
        .collect()
        .get_column("no")
    )


def get_not_null(df: pl.LazyFrame) -> pl.LazyFrame:
    # リストから分解する列
    cols = [
        "ns",
        "lat_left",
        "lat_right",
        "lat_question",
        "lat_left_sign",
        "lat_right_sign",
        "lon_left",
        "lon_right",
        "lon_question",
        "lon_left_sign",
        "lon_right_sign",
    ]
    return (
        df.with_columns(
            # リストを列二つへ分解
            [
                pl.col(col).list.get(n).suffix(f"_{n}")
                for col in cols
                for n in [0, 1]
            ],
        )
        .with_columns(
            # リストの中からnull値以外があれば取り出す
            [pl.coalesce(f"{col}_0", f"{col}_1").alias(col) for col in cols],
        )
        .drop([f"{col}_{n}" for col in cols for n in [0, 1]])
    )


def merge(df: pl.LazyFrame) -> pl.LazyFrame:
    dfl: list[pl.LazyFrame] = []
    for ns in "N", "S":
        df_filterd = df.filter(pl.col("ns").eq(ns))

        with pl.StringCache():
            # 結合対象の番号
            over_no = extract_over_no(df_filterd)

        # 結合対象外
        dfl.append(df_filterd.filter(~pl.col("no").is_in(over_no)))

        # 結合対象
        df_over = df_filterd.filter(pl.col("no").is_in(over_no))
        # 通し番号でまとめ、観測日を計算
        df_over = df_over.groupby("no").agg(
            pl.all().exclude("first", "last"),
            pl.col("first").min(),
            pl.col("last").max(),
        )
        # 上記以外の値を戻す
        df_over = get_not_null(df_over)
        df_over = df_over.with_columns(over=False)

        dfl.append(df_over)

    return pl.concat(dfl)
