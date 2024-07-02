from pathlib import Path

import polars as pl


def filter_df(lf: pl.LazyFrame, expr: pl.Expr) -> pl.DataFrame | None:
    df = lf.filter(expr).sort("no", "ns").collect()
    return df if df.height != 0 else None


def find_null_values_notebook(lf: pl.LazyFrame) -> pl.DataFrame | None:
    """
    欠損値が混入していないか確認する関数
    混入していたもののみ返す
    手帳形式用
    """
    return filter_df(
        lf.select("ns", "no", "lat_left", "lat_right", "first"),
        pl.any_horizontal(pl.all().is_null()),
    )


def find_null_values_report(lf: pl.LazyFrame) -> pl.DataFrame | None:
    """
    欠損値が混入していないか確認する関数
    混入していたもののみ返す
    レポート形式用
    """
    return filter_df(
        lf.select(
            "ns",
            "no",
            "lat_left",
            "lat_right",
            "lon_left",
            "lon_right",
            "first",
            "last",
            "over",
        ),
        pl.any_horizontal(pl.all().is_null()),
    )


def find_observation_reversals(lf: pl.LazyFrame) -> pl.DataFrame | None:
    """
    初観測日時と最終観測日時の関係が逆転していないか確認する関数
    逆転しているもののみ返す
    """
    return filter_df(
        lf.select("ns", "no", "first", "last"),
        pl.col("first") > pl.col("last"),
    )


def find_invalid_obs_range(
    lf: pl.LazyFrame, interval: int
) -> pl.DataFrame | None:
    """
    初観測日時と最終観測日時の差が離れすぎていないか確認する関数
    離れすぎているものを返す
    """
    return filter_df(
        lf.select("ns", "no", "first", "last"),
        (pl.col("last") - pl.col("first")).dt.total_days() > interval,
    )


def find_invalid_lat_range(
    lf: pl.LazyFrame, threthold: int
) -> pl.DataFrame | None:
    """
    緯度の範囲が閾値までに収まってるか確認する関数
    収まっていないもののみ返す
    """
    return filter_df(
        lf.select("ns", "no", "lat_left", "lat_right", "first"),
        ~pl.col("lat_left").is_between(0, threthold)
        | ~pl.col("lat_right").is_between(0, threthold),
    )


def find_invalid_lon_range(lf: pl.LazyFrame) -> pl.DataFrame | None:
    """
    経度の範囲が0~360に収まっているか確認する関数
    収まっていないもののみ返す
    """
    return filter_df(
        lf.select("ns", "no", "lon_left", "lon_right", "first"),
        ~pl.col("lon_left").is_between(0, 360)
        | ~pl.col("lon_right").is_between(0, 360),
    )


def main() -> None:
    df_notebook = pl.scan_parquet(Path("out/ar/notebook_*.parquet"))
    df_report = pl.scan_parquet(Path("out/ar/merged.parquet"))
    df_all = pl.scan_parquet(Path("out/ar/all.parquet"))

    with pl.Config() as cfg, pl.StringCache():
        cfg.set_tbl_cols(-1)
        cfg.set_tbl_rows(-1)

        if (checked := find_null_values_notebook(df_notebook)) is not None:
            print("type notebook null check failed")
            print(checked)
        if (checked := find_null_values_report(df_report)) is not None:
            print("type report null check failed")
            print(checked)
        if (checked := find_observation_reversals(df_all)) is not None:
            print("obs revers check failed")
            print(checked)
        if (checked := find_invalid_obs_range(df_report, 25)) is not None:
            print("obs range check failed")
            print(checked)
        if (checked := find_invalid_lat_range(df_all, 50)) is not None:
            print("lat range check failed")
            print(checked)
        if (checked := find_invalid_lon_range(df_report)) is not None:
            print("lon range check failed")
            print(checked)


if __name__ == "__main__":
    main()
