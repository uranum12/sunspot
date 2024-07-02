import polars as pl


def extract_over_no(df: pl.LazyFrame) -> pl.Series:
    # 複数のシートに分かれていて、データが二つ存在するものの番号
    return (
        df.filter(pl.col("over") & pl.col("no").count().over("no").eq(2))
        .collect()
        .get_column("no")
    )


def get_obs_date(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.group_by("no").agg(
        # 観測日以外
        pl.all().exclude("first", "last"),
        # 観測日の大小を比較
        pl.col("first").min(),
        pl.col("last").max(),
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
                pl.col(col).list.get(n).name.suffix(f"_{n}")
                for col in cols
                for n in [0, 1]
            ]
        )
        .with_columns(
            # リストの中からnull値以外があれば取り出す
            [pl.coalesce(f"{col}_0", f"{col}_1").alias(col) for col in cols]
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
        df_over = get_obs_date(df_over)
        # 観測日以外を戻す
        df_over = get_not_null(df_over)
        # 結合済みとして処理
        df_over = df_over.with_columns(over=False)
        # 使用した列の順番を戻す
        df_over = df_over.select(dfl[-1].collect_schema().names())

        dfl.append(df_over)

    return pl.concat(dfl)
