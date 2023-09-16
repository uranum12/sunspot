import calendar
import sys
from pathlib import Path

import polars as pl

import sn_type


def contains_invalid_chars(col: str, pattern: str) -> pl.Expr:
    return pl.col(col).str.count_matches(pattern) != 1


def check_raw_jst(df: pl.DataFrame, days: int) -> pl.DataFrame:
    return df.filter(
        ~pl.col("date").cast(pl.UInt8).is_between(1, days)
        | contains_invalid_chars("date", r"^\d{1,2}$")
        | contains_invalid_chars("time", r"^\d{1,2}:\d{2}$")
        | contains_invalid_chars("ng", r"^\d{1,2}$")
        | contains_invalid_chars("nf", r"^\d{1,3}$")
        | contains_invalid_chars("sg", r"^\d{1,2}$")
        | contains_invalid_chars("sf", r"^\d{1,3}$"),
    )


def check_raw_ut(df: pl.DataFrame, days: int) -> pl.DataFrame:
    return df.filter(
        ~pl.col("date").cast(pl.UInt8).is_between(1, days)
        | contains_invalid_chars("date", r"^\d{1,2}$")
        | contains_invalid_chars("time", r"^(-\d{1,2}|m?\d{1,2}:\d{2})$")
        | contains_invalid_chars("ng", r"^\d{1,2}$")
        | contains_invalid_chars("nf", r"^\d{1,3}$")
        | contains_invalid_chars("sg", r"^\d{1,2}$")
        | contains_invalid_chars("sf", r"^\d{1,3}$"),
    )


def main() -> int:
    sn_path = Path("data/fujimori_sn")

    for path in sn_path.glob("*-*.csv"):
        year, month = map(int, path.stem.split("-"))
        try:
            file = pl.read_csv(path, infer_schema_length=0)
        except pl.ComputeError as e:
            print(f"{year}/{month}")
            print(e)
            return 1

        if (columns := file.columns) != [
            "date",
            "time",
            "ng",
            "nf",
            "sg",
            "sf",
            "remarks",
        ]:
            print(f"{year}/{month}")
            print(f"invalid columns: {columns}")
            return 1

        match sn_type.detect_time_type(year, month):
            case sn_type.TimeType.JST:
                raw_checked = check_raw_jst(
                    file,
                    calendar.monthrange(year, month)[1],
                )
            case sn_type.TimeType.UT:
                raw_checked = check_raw_ut(
                    file,
                    calendar.monthrange(year, month)[1],
                )
            case _:
                print(f"Err: not supported date for {year}/{month}")
                continue
        if raw_checked.height != 0:
            print(f"{year}/{month}")
            print(raw_checked)
            return 1

    return 0


if __name__ == "__main__":
    ret = main()
    sys.exit(ret)
