from datetime import date
from io import StringIO

import polars as pl
import pytest

import convert_ar_to_butter


@pytest.mark.parametrize(
    ("in_ns", "in_lat_left", "in_lat_right", "out_lat_left", "out_lat_right"),
    [
        ("N", 12, 12, 12, 12),
        ("N", 12, 15, 12, 15),
        ("S", 12, 12, -12, -12),
        ("S", 12, 15, -12, -15),
    ],
)
def test_reverse_south(
    in_ns: str,
    in_lat_left: int,
    in_lat_right: int,
    out_lat_left: int,
    out_lat_right: int,
) -> None:
    df_in = pl.LazyFrame(
        {
            "ns": [in_ns],
            "lat_left": [in_lat_left],
            "lat_right": [in_lat_right],
        },
        schema={
            "ns": pl.Categorical,
            "lat_left": pl.Int8,
            "lat_right": pl.Int8,
        },
    )
    df_out = convert_ar_to_butter.reverse_south(df_in).collect()
    assert df_out.item(0, "lat_left") == out_lat_left
    assert df_out.item(0, "lat_right") == out_lat_right


@pytest.mark.parametrize(
    (
        "in_lat_left",
        "in_lat_right",
        "in_lat_left_sign",
        "in_lat_right_sign",
        "out_lat_left",
        "out_lat_right",
    ),
    [
        (12, 12, None, None, 12, 12),
        (12, 12, "+", "-", 12, -12),
        (12, 15, "-", "+", -12, 15),
    ],
)
def test_reverse_minus(
    in_lat_left: int,
    in_lat_right: int,
    in_lat_left_sign: str | None,
    in_lat_right_sign: str | None,
    out_lat_left: int,
    out_lat_right: int,
) -> None:
    df_in = pl.LazyFrame(
        {
            "lat_left": [in_lat_left],
            "lat_right": [in_lat_right],
            "lat_left_sign": [in_lat_left_sign],
            "lat_right_sign": [in_lat_right_sign],
        },
        schema={
            "lat_left": pl.Int8,
            "lat_right": pl.Int8,
            "lat_left_sign": pl.Categorical,
            "lat_right_sign": pl.Categorical,
        },
    )
    df_out = convert_ar_to_butter.reverse_minus(df_in).collect()
    assert df_out.item(0, "lat_left") == out_lat_left
    assert df_out.item(0, "lat_right") == out_lat_right


@pytest.mark.parametrize(
    ("in_lat_left", "in_lat_right", "out_lat_left", "out_lat_right"),
    [
        (12, 12, 12, 12),
        (12, 15, 12, 15),
        (15, 12, 12, 15),
        (-2, -5, -5, -2),
        (-3, 3, -3, 3),
        (4, -3, -3, 4),
    ],
)
def test_fix_order(
    in_lat_left: int,
    in_lat_right: int,
    out_lat_left: int,
    out_lat_right: int,
) -> None:
    df_in = pl.LazyFrame(
        {
            "lat_left": [in_lat_left],
            "lat_right": [in_lat_right],
        },
        schema={
            "lat_left": pl.Int8,
            "lat_right": pl.Int8,
        },
    )
    df_out = convert_ar_to_butter.fix_order(df_in).collect()
    assert df_out.item(0, "lat_left") == out_lat_left
    assert df_out.item(0, "lat_right") == out_lat_right


@pytest.mark.parametrize(
    (
        "in_first",
        "in_last",
        "out_first_year",
        "out_first_month",
        "out_last_year",
        "out_last_month",
    ),
    [
        (date(1960, 12, 25), date(1961, 1, 7), 1960, 12, 1961, 1),
        (date(2000, 4, 4), date(2000, 4, 4), 2000, 4, 2000, 4),
        (date(2010, 11, 11), date(2010, 12, 3), 2010, 11, 2010, 12),
    ],
)
def test_extract_date(
    in_first: date,
    in_last: date,
    out_first_year: int,
    out_first_month: int,
    out_last_year: int,
    out_last_month: int,
) -> None:
    df_in = pl.LazyFrame(
        {
            "first": [in_first],
            "last": [in_last],
        },
        schema={
            "first": pl.Date,
            "last": pl.Date,
        },
    )
    df_out = convert_ar_to_butter.extract_date(df_in).collect()
    assert df_out.item(0, "first_year") == out_first_year
    assert df_out.item(0, "first_month") == out_first_month
    assert df_out.item(0, "last_year") == out_last_year
    assert df_out.item(0, "last_month") == out_last_month


@pytest.mark.parametrize(
    (
        "in_date",
        "in_first_year",
        "in_first_month",
        "in_last_year",
        "in_last_month",
        "in_lat_left",
        "in_lat_right",
        "out_lat_left",
        "out_lat_right",
    ),
    [
        (
            date(1960, 1, 1),
            [1959, 1959, 1960, 1960, 1960],
            [12, 12, 1, 1, 2],
            [1959, 1960, 1960, 1960, 1960],
            [12, 1, 1, 2, 2],
            [12, -12, 15, -3, 4],
            [15, -4, 18, 5, 4],
            [-12, -3, 15],
            [-4, 5, 18],
        ),
        (
            date(2000, 12, 1),
            [2000, 2000, 2000, 2000, 2001],
            [11, 12, 12, 12, 1],
            [2000, 2000, 2000, 2001, 20001],
            [12, 12, 12, 1, 1],
            [12, -4, 8, -15, 23],
            [15, 5, 8, -12, 24],
            [-15, -4, 8, 12],
            [-12, 5, 8, 15],
        ),
        (
            date(2010, 3, 1),
            [2010, 2010, 2010, 2010, 2010, 2010, 2010, 2010],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [2010, 2010, 2010, 2010, 2010, 2010, 2010, 2010],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [4, 0, 23, -14, -4, 12, 0, 12],
            [4, 0, 23, -12, 0, 13, 4, 12],
            [-14, -4, 0, 0, 4, 12, 12, 23],
            [-12, 0, 0, 4, 4, 12, 13, 23],
        ),
    ],
)
def test_filter_data(
    in_date: date,
    in_first_year: list[int],
    in_first_month: list[int],
    in_last_year: list[int],
    in_last_month: list[int],
    in_lat_left: list[int],
    in_lat_right: list[int],
    out_lat_left: list[int],
    out_lat_right: list[int],
) -> None:
    df_in = pl.LazyFrame(
        {
            "first_year": in_first_year,
            "first_month": in_first_month,
            "last_year": in_last_year,
            "last_month": in_last_month,
            "lat_left": in_lat_left,
            "lat_right": in_lat_right,
        },
        schema={
            "first_year": pl.Int32,
            "first_month": pl.UInt32,
            "last_year": pl.Int32,
            "last_month": pl.UInt32,
            "lat_left": pl.Int8,
            "lat_right": pl.Int8,
        },
    )
    df_out = convert_ar_to_butter.filter_data(df_in, in_date).collect()
    assert df_out.get_column("lat_left").to_list() == out_lat_left
    assert df_out.get_column("lat_right").to_list() == out_lat_right


@pytest.mark.parametrize(
    ("in_lat_left", "in_lat_right", "out_min", "out_max"),
    [
        (
            [12, 15, 23],
            [12, 15, 23],
            [12, 15, 23],
            [12, 15, 23],
        ),
        (
            [12, 14, 23, 24, 30, 31],
            [15, 18, 25, 25, 35, 32],
            [12, 23, 30],
            [18, 25, 35],
        ),
        (
            [12, 13, 13, 23, 25, 26],
            [13, 13, 15, 25, 26, 27],
            [12, 13, 13, 23, 25, 26],
            [13, 13, 15, 25, 26, 27],
        ),
    ],
)
def test_merge_data(
    in_lat_left: list[int],
    in_lat_right: list[int],
    out_min: list[int],
    out_max: list[int],
) -> None:
    df_in = pl.DataFrame(
        {
            "lat_left": in_lat_left,
            "lat_right": in_lat_right,
        },
        schema={
            "lat_left": pl.Int8,
            "lat_right": pl.Int8,
        },
    )
    df_out = convert_ar_to_butter.merge_data(df_in)
    assert df_out.get_column("min").to_list() == out_min
    assert df_out.get_column("max").to_list() == out_max


@pytest.mark.parametrize(
    ("in_min", "in_max", "out_n_min", "out_n_max", "out_s_min", "out_s_max"),
    [
        (
            [-10, -6, 4, 8],
            [-8, -5, 6, 11],
            [4, 8],
            [6, 11],
            [5, 8],
            [6, 10],
        ),
        (
            [-11, -6, 10],
            [-11, 5, 10],
            [0, 10],
            [5, 10],
            [0, 11],
            [6, 11],
        ),
        (
            [0, 5],
            [2, 7],
            [0, 5],
            [2, 7],
            [0],
            [0],
        ),
        (
            [-10],
            [0],
            [0],
            [0],
            [0],
            [10],
        ),
        (
            [3],
            [10],
            [3],
            [10],
            [],
            [],
        ),
        (
            [-10],
            [-3],
            [],
            [],
            [3],
            [10],
        ),
    ],
)
def test_split_data(
    in_min: list[int],
    in_max: list[int],
    out_n_min: list[int],
    out_n_max: list[int],
    out_s_min: list[int],
    out_s_max: list[int],
) -> None:
    df_in = pl.DataFrame(
        {
            "max": in_max,
            "min": in_min,
        },
        schema={
            "max": pl.Int8,
            "min": pl.Int8,
        },
    )
    df_out_n, df_out_s = convert_ar_to_butter.split_data(df_in)
    assert df_out_n.get_column("min").to_list() == out_n_min
    assert df_out_n.get_column("max").to_list() == out_n_max
    assert df_out_s.get_column("min").to_list() == out_s_min
    assert df_out_s.get_column("max").to_list() == out_s_max


@pytest.mark.parametrize(
    ("in_min", "in_max", "out_str"),
    [
        (
            [12, 18, 23],
            [15, 20, 23],
            "12-15 18-20 23-23",
        ),
        (
            [12],
            [15],
            "12-15",
        ),
        (
            [],
            [],
            "",
        ),
    ],
)
def test_convert_to_str(
    in_min: list[str],
    in_max: list[int],
    out_str: str,
) -> None:
    df_in = pl.DataFrame(
        {
            "min": in_min,
            "max": in_max,
        },
        schema={
            "min": pl.Int8,
            "max": pl.Int8,
        },
    )
    assert convert_ar_to_butter.convert_to_str(df_in) == out_str


@pytest.mark.parametrize(
    ("in_start", "in_end", "out_file"),
    [
        (
            date(1960, 1, 1),
            date(1965, 5, 1),
            "//Data File for Butterfly Diagram\n"
            ">>1960/01-1965/05\n\n"
            "<----data---->\n",
        ),
        (
            date(2000, 1, 1),
            date(2010, 12, 1),
            "//Data File for Butterfly Diagram\n"
            ">>2000/01-2010/12\n\n"
            "<----data---->\n",
        ),
    ],
)
def test_write_header(in_start: date, in_end: date, out_file: str) -> None:
    vfile = StringIO()
    convert_ar_to_butter.write_header(vfile, in_start, in_end)
    assert vfile.getvalue() == out_file


@pytest.mark.parametrize(
    ("in_date", "in_data_n", "in_data_s", "out_file"),
    [
        (
            date(1963, 4, 1),
            "12-15 23-27",
            "2-3",
            "1963/04/N:12-15 23-27\n1963/04/S:2-3\n",
        ),
        (
            date(2010, 11, 1),
            "0-0 1-2 3-4",
            "",
            "2010/11/N:0-0 1-2 3-4\n2010/11/S:\n",
        ),
        (
            date(2000, 1, 1),
            "",
            "",
            "2000/01/N:\n2000/01/S:\n",
        ),
    ],
)
def test_write_data(
    in_date: date,
    in_data_n: str,
    in_data_s: str,
    out_file: str,
) -> None:
    vfile = StringIO()
    convert_ar_to_butter.write_data(vfile, in_date, in_data_n, in_data_s)
    assert vfile.getvalue() == out_file


@pytest.mark.parametrize(
    ("in_first", "in_last", "in_replace", "out_start", "out_end"),
    [
        (
            [date(1956, 4, 5), date(1956, 5, 4)],
            [date(1956, 4, 7), date(1956, 5, 6)],
            False,
            date(1956, 4, 5),
            date(1956, 5, 6),
        ),
        (
            [date(2020, 11, 15), date(2020, 12, 30)],
            [date(2020, 11, 17), date(2021, 1, 3)],
            False,
            date(2020, 11, 15),
            date(2021, 1, 3),
        ),
        (
            [date(2020, 11, 15), date(2020, 12, 30)],
            [None, date(2021, 1, 3)],
            False,
            date(2020, 11, 15),
            date(2021, 1, 3),
        ),
        (
            [date(2020, 11, 15), date(2020, 12, 30)],
            [date(2020, 11, 17), date(2021, 1, 3)],
            True,
            date(2020, 11, 1),
            date(2021, 1, 1),
        ),
    ],
)
def test_calc_start_end(
    in_first: list[date],
    in_last: list[date | None],
    in_replace: bool,
    out_start: date,
    out_end: date,
) -> None:
    df_in = pl.LazyFrame({"first": in_first, "last": in_last})
    start, end = convert_ar_to_butter.calc_start_end(df_in, replace=in_replace)
    assert start == out_start
    assert end == out_end
