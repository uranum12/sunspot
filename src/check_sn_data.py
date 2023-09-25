from pathlib import Path

import polars as pl


def filter_df(lf: pl.LazyFrame, expr: pl.Expr) -> pl.DataFrame | None:
    df = lf.filter(expr).collect()
    return df if df.height != 0 else None


def find_null_values(lf: pl.LazyFrame) -> pl.DataFrame | None:
    """
    nullチェック
    混入しているものを返す
    """
    return filter_df(
        lf,
        pl.any_horizontal(pl.col("time", "ng", "nf", "sg", "sf").is_null())
        & ~pl.all_horizontal(pl.col("time", "ng", "nf", "sg", "sf").is_null())
        | pl.col("date").is_null(),
    )


def find_gf_reversals(lf: pl.LazyFrame) -> pl.DataFrame | None:
    """
    g(黒点群数)とf(黒点数)の関係が逆転していないかを確認する関数
    逆転しているものを返す
    """
    return filter_df(
        lf,
        (pl.col("ng") > pl.col("nf")) | (pl.col("sg") > pl.col("sf")),
    )


def check_by_total(
    lf: pl.LazyFrame,
    index: pl.LazyFrame,
) -> pl.DataFrame | None:
    return (
        df
        if (
            df := lf.select(
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month"),
                (pl.col("nf") + pl.col("ng") * 10).alias("nt"),
                (pl.col("sf") + pl.col("sg") * 10).alias("st"),
            )
            .group_by("year", "month")
            .agg(
                pl.col("nt").sum(),
                pl.col("st").sum(),
            )
            .join(index, on=["year", "month"], how="outer")
            .filter(
                (pl.col("nt") != pl.col("n")) | (pl.col("st") != pl.col("s")),
            )
            .sort("year", "month")
            .collect()
        ).height
        != 0
        else None
    )


def main() -> None:
    file = pl.scan_parquet(Path("out/sn/all.parquet"))
    index = pl.scan_parquet(Path("out/sn/index.parquet"))

    with pl.Config() as cfg:
        cfg.set_tbl_cols(-1)
        cfg.set_tbl_rows(-1)

        if (checked := find_null_values(file)) is not None:
            print("null check failed")
            print(checked)
        if (checked := find_gf_reversals(file)) is not None:
            print("gf reversals check failed")
            print(checked)
        if (checked := check_by_total(file, index)) is not None:
            print("total check failed")
            print(checked)


if __name__ == "__main__":
    main()
