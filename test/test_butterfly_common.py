from datetime import date

import polars as pl
import pytest

import butterfly_common


def test_cast_lat_sign() -> None:
    df_in = pl.LazyFrame(
        {
            "lat_left": [12],
            "lat_right": [12],
        },
        schema={
            "lat_left": pl.UInt8,
            "lat_right": pl.UInt8,
        },
    )
    df_out = butterfly_common.cast_lat_sign(df_in)
    assert df_out.schema == {"lat_left": pl.Int8, "lat_right": pl.Int8}


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
    df_out = butterfly_common.reverse_south(df_in).collect()
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
    df_out = butterfly_common.reverse_minus(df_in).collect()
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
    df_out = butterfly_common.fix_order(df_in).collect()
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
    df_out = butterfly_common.extract_date(df_in).collect()
    assert df_out.item(0, "first_year") == out_first_year
    assert df_out.item(0, "first_month") == out_first_month
    assert df_out.item(0, "last_year") == out_last_year
    assert df_out.item(0, "last_month") == out_last_month


@pytest.mark.parametrize(
    (
        "in_date",
        "in_first",
        "in_last",
        "in_lat_left",
        "in_lat_right",
        "out_lat_left",
        "out_lat_right",
    ),
    [
        (
            date(1954, 5, 5),
            [date(1954, 4, 1), date(1954, 5, 1), date(1954, 5, 1)],
            [None, None, None],
            [12, 23, 34],
            [13, 24, 35],
            [23, 34],
            [24, 35],
        ),
        (
            date(2020, 7, 7),
            [
                date(2020, 6, 29),
                date(2020, 7, 1),
                date(2020, 7, 6),
                date(2020, 7, 7),
                date(2020, 7, 8),
            ],
            [
                date(2020, 7, 8),
                date(2020, 7, 4),
                date(2020, 7, 8),
                date(2020, 7, 9),
                date(2020, 7, 10),
            ],
            [12, 1, 1, 2, 2],
            [12, -12, 15, -3, 4],
            [12, 1, 2],
            [12, 15, -3],
        ),
    ],
)
def test_filter_data_daily(
    in_date: date,
    in_first: list[date],
    in_last: list[date | None],
    in_lat_left: list[int],
    in_lat_right: list[int],
    out_lat_left: list[int],
    out_lat_right: list[int],
) -> None:
    df_in = pl.LazyFrame(
        {
            "first": in_first,
            "last": in_last,
            "lat_left": in_lat_left,
            "lat_right": in_lat_right,
        },
        schema={
            "first": pl.Date,
            "last": pl.Date,
            "lat_left": pl.Int8,
            "lat_right": pl.Int8,
        },
    )
    df_out = butterfly_common.filter_data_daily(df_in, in_date).collect()
    assert (
        df_out.get_column("lat_left").to_list().sort() == out_lat_left.sort()
    )
    assert (
        df_out.get_column("lat_right").to_list().sort() == out_lat_right.sort()
    )


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
def test_filter_data_monthly(
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
    df_out = butterfly_common.filter_data_monthly(df_in, in_date).collect()
    assert sorted(df_out.get_column("lat_left").to_list()) == sorted(
        out_lat_left,
    )
    assert sorted(df_out.get_column("lat_right").to_list()) == sorted(
        out_lat_right,
    )


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
    start, end = butterfly_common.calc_start_end(df_in, replace=in_replace)
    assert start == out_start
    assert end == out_end
