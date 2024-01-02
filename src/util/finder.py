import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl


def main(argv: list[str]) -> None:
    if len(argv) > 2:  # noqa: PLR2004
        match argv[1]:
            case "ar":
                ns = argv[2][0:1].upper()
                no = int(argv[2][1:])
                print(ns, no)
                with pl.Config() as cfg:
                    cfg.set_tbl_rows(-1)
                    print(
                        pl.scan_parquet(Path("out/ar/all.parquet"))
                        .filter(pl.col("ns").eq(ns))
                        .filter(pl.col("no").is_between(no - 1, no + 1))
                        .select("ns", "no", "first", "last")
                        .collect(),
                    )
            case "sn":
                year, month, day = map(int, argv[2].split("-"))
                target_date = date(year, month, day)
                start_date = target_date - timedelta(days=1)
                end_date = target_date + timedelta(days=1)
                print(target_date)
                print(
                    pl.scan_parquet(Path("out/sn/all.parquet"))
                    .filter(pl.col("date").is_between(start_date, end_date))
                    .collect(),
                )


if __name__ == "__main__":
    main(sys.argv)
