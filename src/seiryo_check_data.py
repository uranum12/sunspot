from pathlib import Path

import polars as pl


def find_null_values(df: pl.DataFrame) -> pl.DataFrame:
    """欠損値を検索する

    Args:
        df (pl.DataFrame): 黒点群データまたは黒点数データ

    Returns:
        pl.DataFrame: 欠損値を含む行のみの新しいデータフレーム
    """
    return df.filter(pl.any_horizontal(pl.all().is_null()))


def find_invalid_group_number(df: pl.DataFrame) -> pl.DataFrame:
    """不正なグループ番号を検索する

    Args:
        df (pl.DataFrame): 黒点群データ

    Returns:
        pl.DataFrame: 不正値を含む行のみの新しいデータフレーム
    """
    return (
        df.lazy()
        .group_by("date")
        .agg(
            pl.col("no").alias("original"),
            pl.col("no")
            .count()
            .map_elements(lambda no: list(range(1, no + 1)))
            .cast(pl.UInt8)
            .alias("expected"),
        )
        .with_columns(
            pl.col("original", "expected").list.sort(),
        )
        .filter(
            pl.col("original") != pl.col("expected"),
        )
        .collect()
    )


def find_invalid_lat_range(df: pl.DataFrame, threshold: int) -> pl.DataFrame:
    """不正な範囲に存在する緯度を検索する

    Args:
        df (pl.DataFrame): 黒点群データ
        threshold (int): 緯度の閾値

    Returns:
        pl.DataFrame: 不正値を含む行のみの新しいデータフレーム
    """
    return df.filter(
        pl.any_horizontal(
            ~pl.col("lat_min").abs().is_between(0, threshold),
            ~pl.col("lat_max").abs().is_between(0, threshold),
        ),
    )


def find_invalid_lon_range(
    df: pl.DataFrame,
    min_threshold: int,
    max_threshold: int,
) -> pl.DataFrame:
    """不正な範囲に存在する経度を検索する

    Args:
        df (pl.DataFrame): 黒点群データ
        min_threshold (int): 経度の閾値の最小
        max_threshold (int): 経度の閾値の最大

    Returns:
        pl.DataFrame: 不正値を含む行のみの新しいデータフレーム
    """
    return df.filter(
        pl.any_horizontal(
            ~pl.col("lon_min").is_between(min_threshold, max_threshold),
            ~pl.col("lon_max").is_between(min_threshold, max_threshold),
        ),
    )


def find_invalid_lat_interval(df: pl.DataFrame, interval: int) -> pl.DataFrame:
    """不正な間隔を持つ緯度を検索する

    Args:
        df (pl.DataFrame): 黒点群データ
        interval (int): 緯度の間隔の最大値

    Returns:
        pl.DataFrame: 不正値を含む行のみの新しいデータフレーム
    """
    return df.with_columns(
        (pl.col("lat_max") - pl.col("lat_min")).alias("interval"),
    ).filter(
        pl.col("interval") > interval,
    )


def find_invalid_lon_interval(df: pl.DataFrame, interval: int) -> pl.DataFrame:
    """不正な間隔を持つ経度を検索する

    Args:
        df (pl.DataFrame): 黒点群データ
        interval (int): 経度の間隔の最大値

    Returns:
        pl.DataFrame: 不正値を含む行のみの新しいデータフレーム
    """
    return df.with_columns(
        (pl.col("lon_max") - pl.col("lon_min")).alias("interval"),
    ).filter(
        pl.col("interval") > interval,
    )


def find_duplicate_date(df: pl.DataFrame) -> pl.DataFrame:
    """重複した日付を検索する

    Args:
        df (pl.DataFrame): 黒点数データ

    Returns:
        pl.DataFrame: 重複した日付を含む行のみの新しいデータフレーム
    """
    return df.filter(pl.col("date").is_duplicated())


def find_invalid_total_group(df: pl.DataFrame) -> pl.DataFrame:
    """不正な黒点群数の合計値を検索する

    Args:
        df (pl.DataFrame): 黒点数データ

    Returns:
        pl.DataFrame: 不正値を含む行のみの新しいデータフレーム
    """
    return df.with_columns(
        pl.col("ng").add(pl.col("sg")).alias("nsg"),
    ).filter(
        pl.col("tg") != pl.col("nsg"),
    )


def find_invalid_total_number(df: pl.DataFrame) -> pl.DataFrame:
    """不正な黒点数の合計値を検索する

    Args:
        df (pl.DataFrame): 黒点数データ

    Returns:
        pl.DataFrame: 不正値を含む行のみの新しいデータフレーム
    """
    return df.with_columns(
        pl.col("nf").add(pl.col("sf")).alias("nsf"),
    ).filter(
        pl.col("tf") != pl.col("nsf"),
    )


def check_ar_data(df: pl.DataFrame) -> None:
    lat_threshold = 50
    lon_min_threshold = -180
    lon_max_threshold = 180
    lat_interval = 15
    lon_interval = 30

    ret = find_null_values(df)
    if ret.height != 0:
        print("Null values found in AR data")
        print(ret.sort("date", "no"))

    ret = find_invalid_group_number(df)
    if ret.height != 0:
        print("Invalid group numbers found in AR data")
        print(ret.sort("date"))

    ret = find_invalid_lat_range(df, lat_threshold)
    if ret.height != 0:
        print("Invalid latitude range values found in AR data")
        print(ret.drop("lon_min", "lon_max").sort("date", "no"))

    ret = find_invalid_lon_range(df, lon_min_threshold, lon_max_threshold)
    if ret.height != 0:
        print("Invalid longitude range values found in AR data")
        print(ret.drop("lat_min", "lat_max").sort("date", "no"))

    ret = find_invalid_lat_interval(df, lat_interval)
    if ret.height != 0:
        print("Invalid latitude interval found in AR data")
        print(ret.drop("lon_min", "lon_max").sort("date", "no"))

    ret = find_invalid_lon_interval(df, lon_interval)
    if ret.height != 0:
        print("Invalid longitude interval found in AR data")
        print(ret.drop("lat_min", "lat_max").sort("date", "no"))


def check_sn_data(df: pl.DataFrame) -> None:
    ret = find_null_values(df)
    if ret.height != 0:
        print("Null values found in SN data")
        print(ret.sort("date"))

    ret = find_duplicate_date(df)
    if ret.height != 0:
        print("Duplicate dates found in SN data")
        print(ret.sort("date"))

    ret = find_invalid_total_group(df)
    if ret.height != 0:
        print("Invalid total group number found in SN data")
        print(ret.drop("nf", "sf", "tf").sort("date"))

    ret = find_invalid_total_number(df)
    if ret.height != 0:
        print("Invalid total number found in SN data")
        print(ret.drop("ng", "sg", "tg").sort("date"))


def main() -> None:
    df_ar = pl.read_parquet(Path("out/seiryo/ar.parquet"))
    df_sn = pl.read_parquet(Path("out/seiryo/sn.parquet"))

    with pl.Config() as cfg:
        cfg.set_tbl_cols(-1)
        cfg.set_tbl_rows(-1)

        check_ar_data(df_ar)
        check_sn_data(df_sn)


if __name__ == "__main__":
    main()
