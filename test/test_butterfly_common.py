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


def test_drop_lat_null() -> None:
    df_in = pl.LazyFrame(
        {
            "lat_left": [4, None],
            "lat_right": [12, None],
        },
        schema={
            "lat_left": pl.Int8,
            "lat_right": pl.Int8,
        },
    )
    df_out = butterfly_common.drop_lat_null(df_in).collect()
    assert df_out.get_column("lat_left").to_list() == [4]
    assert df_out.get_column("lat_right").to_list() == [12]


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
    ("in_lat_left", "in_lat_right", "out_lat_min", "out_lat_max"),
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
    out_lat_min: int,
    out_lat_max: int,
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
    assert df_out.item(0, "lat_min") == out_lat_min
    assert df_out.item(0, "lat_max") == out_lat_max


@pytest.mark.parametrize(
    ("in_first", "in_last", "out_last"),
    [
        (date(1956, 8, 1), None, date(1956, 8, 31)),
        (date(2020, 5, 5), date(2020, 5, 7), date(2020, 5, 7)),
        (date(1964, 2, 1), None, date(1964, 2, 29)),
        (date(1965, 2, 1), None, date(1965, 2, 28)),
    ],
)
def test_complement_last(
    in_first: date,
    in_last: date | None,
    out_last: date,
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
    df_out = butterfly_common.complement_last(df_in).collect()
    assert df_out.item(0, "first") == in_first
    assert df_out.item(0, "last") == out_last


@pytest.mark.parametrize(
    ("in_first", "in_last", "out_first", "out_last"),
    [
        (
            date(1956, 5, 5),
            date(1956, 5, 30),
            date(1956, 5, 1),
            date(1956, 5, 1),
        ),
        (
            date(2020, 8, 30),
            date(2020, 9, 5),
            date(2020, 8, 1),
            date(2020, 9, 1),
        ),
    ],
)
def test_truncate_day(
    in_first: date,
    in_last: date,
    out_first: date,
    out_last: date,
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
    df_out = butterfly_common.truncate_day(df_in).collect()
    assert df_out.item(0, "first") == out_first
    assert df_out.item(0, "last") == out_last


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
def test_filter_data(
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
    df_out = butterfly_common.filter_data(df_in, in_date).collect()
    assert (
        df_out.get_column("lat_left").to_list().sort() == out_lat_left.sort()
    )
    assert (
        df_out.get_column("lat_right").to_list().sort() == out_lat_right.sort()
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
