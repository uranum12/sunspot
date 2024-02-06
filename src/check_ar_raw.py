import sys
from functools import reduce
from pathlib import Path

import polars as pl

import ar_type

columns: dict[ar_type.SchemaType, list[str]] = {
    ar_type.SchemaType.NOTEBOOK_1: ["no", "ns", "lat"],
    ar_type.SchemaType.NOTEBOOK_2: ["no", "lat"],
    ar_type.SchemaType.NOTEBOOK_3: ["no", "lat"],
    ar_type.SchemaType.OLD: ["no", "lat", "lon", "first", "last"],
    ar_type.SchemaType.NEW: ["no", "lat", "lon", "first", "last"],
}

pat_months = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"

patterns: dict[ar_type.SchemaType, dict[str, str]] = {
    ar_type.SchemaType.NOTEBOOK_1: {
        "ns": r"^[NS]$",
        "no": r"^\d{1,4}$",
        "lat": r"^\d{1,2}(~\d{1,2})?$",
    },
    ar_type.SchemaType.NOTEBOOK_2: {
        "no": r"^[NS]\d{1,3}$",
        "lat": r"^\d{1,2}(~\d{1,2})?\??$",
    },
    ar_type.SchemaType.NOTEBOOK_3: {
        "no": r"^[NS]\d{4}_\d{1,2}$",
        "lat": r"^-?\d{1,2}(~-?\d{1,2})?$",
    },
    ar_type.SchemaType.OLD: {
        "no": r"^[NS]\d{4}$",
        "lat": r"(^/$|^[p-]?\d{1,2}(~[p-]?\d{1,2})?\??$)",
        "lon": r"(^/$|^[p-]?\d{1,3}(~[p-]?\d{1,3})?\??$)",
        "first": r"^\d{1,2}$",
        "last": r"^\d{1,2}$",
    },
    ar_type.SchemaType.NEW: {
        "no": r"^[NS]\d{4}$",
        "lat": r"(^/$|^[p-]?\d{1,2}(~[p-]?\d{1,2})?\??$)",
        "lon": r"(^/$|^[p-]?\d{1,3}(~[p-]?\d{1,3})?\??$)",
        "first": rf"^{pat_months}\.\d{{1,2}}$",
        "last": rf"^{pat_months}\.\d{{1,2}}$",
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
        )
    )


def check_file(path: Path, cols: list[str], pats: dict[str, str]) -> None:
    file = pl.read_csv(path, infer_schema_length=0)

    if (c := file.columns) != cols:
        msg = f"invalid columns: {c}"
        raise ValueError(msg)

    if (raw_checked := check_raw(file, pats)).height != 0:
        raise ValueError(raw_checked)


def main() -> int:
    ar_path = Path("data/fujimori_ar")

    for path in ar_path.glob("*-*.csv"):
        year, month = map(int, path.stem.split("-"))
        if schema_type := ar_type.detect_schema_type(year, month):
            try:
                check_file(path, columns[schema_type], patterns[schema_type])
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
