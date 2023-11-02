from pathlib import Path
from pprint import pprint

import polars as pl


def calc_wolf_number(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        (pl.col("ng") + pl.col("sg")).alias("tg"),
        (pl.col("nf") + pl.col("sf")).alias("tf"),
    ).with_columns(
        # R = 10g + f
        (pl.col("ng").cast(pl.UInt16) * 10 + pl.col("nf")).alias("nr"),
        (pl.col("sg").cast(pl.UInt16) * 10 + pl.col("sf")).alias("sr"),
        (pl.col("tg").cast(pl.UInt16) * 10 + pl.col("tf")).alias("tr"),
    )


def agg_monthly(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.with_columns(pl.col("date").dt.truncate("1mo"))
        .group_by("date")
        .mean()
    )


def main() -> None:
    data_file = Path("out/sn/all.parquet")
    output_path = Path("out/wolf")
    output_path.mkdir(parents=True, exist_ok=True)

    df = (
        pl.scan_parquet(data_file)
        .drop("time", "remarks")
        .drop_nulls()
        .pipe(calc_wolf_number)
        .collect()
    )

    print("=== daily ===")
    pprint(df.schema)
    print(df)
    df.write_parquet(output_path / "fujimori.parquet")

    df = df.lazy().drop("time").pipe(agg_monthly).sort("date").collect()

    print("=== monthly ===")
    pprint(df.schema)
    print(df)
    df.write_parquet(output_path / "fujimori_monthly.parquet")


if __name__ == "__main__":
    main()
