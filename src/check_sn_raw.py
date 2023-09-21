import calendar
import sys
from functools import reduce
from pathlib import Path

import polars as pl

import sn_type

patterns: dict[sn_type.TimeType, dict[str, str]] = {
    sn_type.TimeType.JST: {
        "date": r"^\d{1,2}$",
        "time": r"^\d{1,2}:\d{2}$",
        "ng": r"^\d{1,2}$",
        "nf": r"^\d{1,3}$",
        "sg": r"^\d{1,2}$",
        "sf": r"^\d{1,3}$",
    },
    sn_type.TimeType.UT: {
        "date": r"^\d{1,2}$",
        "time": r"^(-\d{1,2}|m?\d{1,2}:\d{2})$",
        "ng": r"^\d{1,2}$",
        "nf": r"^\d{1,3}$",
        "sg": r"^\d{1,2}$",
        "sf": r"^\d{1,3}$",
    },
}


def check_raw(df: pl.DataFrame, pats: dict[str, str]) -> pl.DataFrame:
    return df.filter(
        reduce(
            lambda acc, cur: acc | cur,
            [
                pl.col(col).str.count_matches(pat) != 1
                for col, pat in pats.items()
            ],
        ),
    )


def check_date(df: pl.DataFrame, days: int) -> pl.DataFrame:
    return df.filter(~pl.col("date").cast(pl.UInt8).is_between(1, days))


def check_file(path: Path, days: int, pats: dict[str, str]) -> None:
    file = pl.read_csv(path, infer_schema_length=0)

    if (c := file.columns) != [
        "date",
        "time",
        "ng",
        "nf",
        "sg",
        "sf",
        "remarks",
    ]:
        msg = f"invalid columns: {c}"
        raise ValueError(msg)

    if (date_checked := check_date(file, days)).height != 0:
        raise ValueError(date_checked)

    if (raw_checked := check_raw(file, pats)).height != 0:
        raise ValueError(raw_checked)


def main() -> int:
    sn_path = Path("data/fujimori_sn")

    for path in sn_path.glob("*-*.csv"):
        year, month = map(int, path.stem.split("-"))
        if time_type := sn_type.detect_time_type(year, month):
            try:
                days = calendar.monthrange(year, month)[1]
                check_file(path, days, patterns[time_type])
            except (pl.ComputeError, ValueError) as e:
                print(f"{year}/{month}")
                print(e)
                return 1
        else:
            print(f"Err: not supported date for {year}/{month}")
            continue

    return 0


if __name__ == "__main__":
    ret = main()
    sys.exit(ret)
