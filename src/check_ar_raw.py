import sys
from pathlib import Path

import polars as pl

import ar_type


def contains_invalid_chars(col: str, pattern: str) -> pl.Expr:
    return pl.col(col).str.count_matches(pattern) != 1


def check_raw_notebook(df: pl.DataFrame) -> pl.DataFrame:
    """
    使用できない文字が混入していないかを確認する関数
    全て文字列として読み込んだものを使用すること
    手帳形式専用
    混入していたものを返す
    """
    return df.filter(
        contains_invalid_chars("ns", r"^[NS]$")
        | contains_invalid_chars("no", r"^\d{1,4}(\.\d)?$")
        | contains_invalid_chars("lat", r"^-?\d{1,2}(~-?\d{1,2})?\??$"),
    )


def check_raw_old(df: pl.DataFrame) -> pl.DataFrame:
    """
    使用できない文字が混入していないかを確認する関数
    全て文字列として読み込んだものを使用すること
    古い形式専用
    混入していたものを返す
    """
    return df.filter(
        contains_invalid_chars("No", r"^[NS]\d{4}$")
        | contains_invalid_chars(
            "lat",
            r"(^/$|^[p-]?\d{1,2}(~[p-]?\d{1,2})?\??$)",
        )
        | contains_invalid_chars(
            "lon",
            r"(^/$|^[p-]?\d{1,3}(~[p-]?\d{1,3})?\??$)",
        )
        | contains_invalid_chars("first", r"^\d{1,2}$")
        | contains_invalid_chars("last", r"^\d{1,2}$"),
    )


def check_raw_new(df: pl.DataFrame) -> pl.DataFrame:
    """
    使用できない文字が混入してないかを確認する関数
    全て文字列として読み込んだものを使用すること
    新しい形式専用
    混入していたものを返す
    """
    return df.filter(
        contains_invalid_chars("No", r"^[NS]\d{4}$")
        | contains_invalid_chars(
            "lat",
            r"(^/$|^[p-]?\d{1,2}(~[p-]?\d{1,2})?\??$)",
        )
        | contains_invalid_chars(
            "lon",
            r"(^/$|^[p-]?\d{1,3}(~[p-]?\d{1,3})?\??$)",
        )
        | contains_invalid_chars(
            "first",
            r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.\d{1,2}$",
        )
        | contains_invalid_chars(
            "last",
            r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.\d{1,2}$",
        ),
    )


def main() -> int:
    ar_path = Path("data/fujimori_ar")

    for path in ar_path.glob("*-*.csv"):
        year, month = map(int, path.stem.split("-"))
        try:
            file = pl.read_csv(path, infer_schema_length=0)
        except pl.ComputeError as e:
            print(f"{year}/{month}")
            print(e)
            return 1

        match ar_type.detect_schema_type(year, month):
            case ar_type.SchemaType.NOTEBOOK:
                # 手帳形式
                if (columns := file.columns) != ["no", "ns", "lat"]:
                    print(f"{year}/{month}")
                    print(f"invalid columns: {columns}")
                    return 1
            case ar_type.SchemaType.OLD | ar_type.SchemaType.NEW:
                # 古い形式と新しい形式
                if (columns := file.columns) != [
                    "No",
                    "lat",
                    "lon",
                    "first",
                    "last",
                ]:
                    print(f"{year}/{month}")
                    print(f"invalid columns: {columns}")
                    return 1
            case _:
                print(f"Err: not supported date for {year}/{month}")
                continue

        match ar_type.detect_schema_type(year, month):
            case ar_type.SchemaType.NOTEBOOK:
                # 手帳形式
                raw_checked = check_raw_notebook(file)
            case ar_type.SchemaType.OLD:
                # 古い形式
                raw_checked = check_raw_old(file)
            case ar_type.SchemaType.NEW:
                # 新しい形式
                raw_checked = check_raw_new(file)
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
