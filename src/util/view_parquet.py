import sys
from pathlib import Path

import polars as pl


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        path = Path(argv[1])
        print(
            pl.scan_parquet(path)
            .filter(~pl.all_horizontal(pl.all().is_null()))
            .collect()
            .glimpse(return_as_string=True),
        )


if __name__ == "__main__":
    main(sys.argv)
