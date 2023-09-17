import polars as pl


def calc_obs_date(df: pl.LazyFrame, year: int, month: int) -> pl.LazyFrame:
    return df.with_columns(
        # 初観測日、最終観測日に年と月を付与
        [
            pl.date(year, month, pl.col(obs_time)).alias(obs_time)
            for obs_time in ["first", "last"]
        ],
    )
