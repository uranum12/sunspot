from pathlib import Path
from pprint import pprint

import polars as pl


def main() -> None:
    data_file = Path("out/sn/all.parquet")
    output_path = Path("out/wolf")
    output_path.mkdir(parents=True, exist_ok=True)

    df = (
        pl.scan_parquet(data_file)
        .drop("remarks")
        .with_columns(
            pl.col("ng").add(pl.col("sg")).alias("tg"),
            pl.col("nf").add(pl.col("sf")).alias("tf"),
        )
        .with_columns(
            pl.col("nf").add(pl.col("ng") * 10).alias("nr"),
            pl.col("sf").add(pl.col("sg") * 10).alias("sr"),
            pl.col("tf").add(pl.col("tg") * 10).alias("tr"),
        )
        .collect()
    )

    print("=== daily ===")
    pprint(df.schema)
    print(df)
    df.write_parquet(output_path / "fujimori.parquet")

    df = (
        df.lazy()
        .with_columns(pl.col("date").dt.truncate("1mo"))
        .drop("time")
        .group_by("date")
        .mean()
        .sort("date")
        .collect()
    )

    print("=== monthly ===")
    pprint(df.schema)
    print(df)
    df.write_parquet(output_path / "fujimori_monthly.parquet")


if __name__ == "__main__":
    main()
