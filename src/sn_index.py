from pathlib import Path

import polars as pl


def main() -> None:
    data_path = Path("data/fujimori_sn_index/*.csv")
    output_path = Path("out/sn/index.parquet")

    pl.scan_csv(data_path).with_columns(
        pl.col("year").cast(pl.Int32), pl.col("month").cast(pl.Int8)
    ).sink_parquet(output_path)


if __name__ == "__main__":
    main()
