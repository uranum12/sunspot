import polars as pl


def calc_obs_date(df: pl.LazyFrame, year: int, month: int) -> pl.LazyFrame:
    return (
        # 観測日を月と日に分離
        # この時月名を英語から数字へ変換
        df.with_columns(pl.col("first", "last").str.split("."))
        .with_columns(
            pl.col("first", "last")
            .list.get(0)
            .replace(
                {
                    "jan": 1,
                    "feb": 2,
                    "mar": 3,
                    "apr": 4,
                    "may": 5,
                    "jun": 6,
                    "jul": 7,
                    "aug": 8,
                    "sep": 9,
                    "oct": 10,
                    "nov": 11,
                    "dec": 12,
                },
                default=None,
                return_dtype=pl.UInt8,
            )
            .name.suffix("_month"),
            pl.col("first", "last")
            .list.get(1)
            .cast(pl.UInt8)
            .name.suffix("_day"),
        )
        .with_columns(
            # 12月で翌年の1月までが範囲の場合、年をひとつ加算
            [
                pl.when(
                    (month == 12) & (pl.col(f"{obs_time}_month") == 1)  # noqa: PLR2004
                )
                .then(
                    pl.date(
                        year + 1,
                        pl.col(f"{obs_time}_month"),
                        pl.col(f"{obs_time}_day"),
                    )
                )
                .otherwise(
                    pl.date(
                        year,
                        pl.col(f"{obs_time}_month"),
                        pl.col(f"{obs_time}_day"),
                    )
                )
                .alias(obs_time)
                for obs_time in ["first", "last"]
            ]
        )
        .drop("first_month", "first_day", "last_month", "last_day")
    )
