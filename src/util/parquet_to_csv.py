from pathlib import Path

import polars as pl


def main() -> None:
    out_path = Path("out")

    for path in out_path.rglob("*.parquet"):
        file = pl.read_parquet(path)
        file.write_csv(path.with_suffix(".csv"))


if __name__ == "__main__":
    main()
