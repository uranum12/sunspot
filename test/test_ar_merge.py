from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

import ar_merge


@pytest.mark.parametrize(
    ("in_no", "in_over", "out_no"),
    [
        ([1, 2, 3, 4], [False, False, False, False], []),
        ([1, 2, 3, 3], [False, False, True, False], [3]),
        ([1, 2, 3, 3], [False, False, False, False], []),
        ([1, 2, 2, 2], [False, True, False, False], []),
        ([1, 2, 3, 4], [True, True, False, False], []),
    ],
)
def test_extract_over_no(
    in_no: list[int], in_over: list[int], out_no: list[int]
) -> None:
    df_in = pl.LazyFrame(
        {"no": in_no, "over": in_over},
        schema={"no": pl.UInt32, "over": pl.Boolean},
    )
    s_expected = pl.Series("no", out_no, dtype=pl.UInt32)
    s_out = ar_merge.extract_over_no(df_in)
    assert_series_equal(s_out, s_expected)


@pytest.mark.parametrize(
    ("in_no", "in_first", "in_last", "out_no", "out_first", "out_last"),
    [
        (
            [1, 1],
            [date(1980, 7, 30), date(1980, 7, 31)],
            [date(1980, 8, 1), date(1980, 8, 3)],
            [1],
            [date(1980, 7, 30)],
            [date(1980, 8, 3)],
        ),
        (
            [1, 1],
            [date(1980, 12, 25), date(1980, 12, 31)],
            [date(1981, 1, 3), date(1981, 1, 3)],
            [1],
            [date(1980, 12, 25)],
            [date(1981, 1, 3)],
        ),
    ],
)
def test_get_obs_date(
    in_no: list[int],
    in_first: list[date],
    in_last: list[date],
    out_no: list[list[int]],
    out_first: list[int],
    out_last: list[int],
) -> None:
    df_in = pl.LazyFrame(
        {"no": in_no, "first": in_first, "last": in_last},
        schema={"no": pl.UInt32, "first": pl.Date, "last": pl.Date},
    )
    df_expected = pl.LazyFrame(
        {"no": out_no, "first": out_first, "last": out_last},
        schema={"no": pl.UInt32, "first": pl.Date, "last": pl.Date},
    )
    df_out = ar_merge.get_obs_date(df_in)
    assert_frame_equal(df_out, df_expected, check_column_order=False)


def test_get_not_null() -> None:
    df_in = pl.LazyFrame(
        {
            "no": [1],
            "ns": [["N", "N"]],
            "lat_left": [[None, 12]],
            "lat_right": [[None, 12]],
            "lat_question": [[None, None]],
            "lat_left_sign": [["+", None]],
            "lat_right_sign": [["-", None]],
            "lon_left": [[123, None]],
            "lon_right": [[132, None]],
            "lon_question": [[None, "?"]],
            "lon_left_sign": [[None, "-"]],
            "lon_right_sign": [[None, None]],
        },
        schema={
            "no": pl.UInt32,
            "ns": pl.List(pl.Utf8),
            "lat_left": pl.List(pl.UInt8),
            "lat_right": pl.List(pl.UInt8),
            "lat_question": pl.List(pl.Utf8),
            "lat_left_sign": pl.List(pl.Utf8),
            "lat_right_sign": pl.List(pl.Utf8),
            "lon_left": pl.List(pl.UInt16),
            "lon_right": pl.List(pl.UInt16),
            "lon_question": pl.List(pl.Utf8),
            "lon_left_sign": pl.List(pl.Utf8),
            "lon_right_sign": pl.List(pl.Utf8),
        },
    )
    df_expected = pl.LazyFrame(
        {
            "no": [1],
            "ns": ["N"],
            "lat_left": [12],
            "lat_right": [12],
            "lat_question": [None],
            "lat_left_sign": ["+"],
            "lat_right_sign": ["-"],
            "lon_left": [123],
            "lon_right": [132],
            "lon_question": ["?"],
            "lon_left_sign": ["-"],
            "lon_right_sign": [None],
        },
        schema={
            "no": pl.UInt32,
            "ns": pl.Utf8,
            "lat_left": pl.UInt8,
            "lat_right": pl.UInt8,
            "lat_question": pl.Utf8,
            "lat_left_sign": pl.Utf8,
            "lat_right_sign": pl.Utf8,
            "lon_left": pl.UInt16,
            "lon_right": pl.UInt16,
            "lon_question": pl.Utf8,
            "lon_left_sign": pl.Utf8,
            "lon_right_sign": pl.Utf8,
        },
    )
    df_out = ar_merge.get_not_null(df_in)
    assert_frame_equal(df_out, df_expected, check_column_order=False)


def test_merge() -> None:
    df_in = pl.LazyFrame(
        {
            "ns": ["N", "N", "N", "S", "S"],
            "no": [1, 2, 2, 1, 1],
            "lat_left": [12, None, 12, 1, None],
            "lat_right": [12, None, 15, 5, None],
            "lat_left_sign": [None, None, None, "-", None],
            "lat_right_sign": [None, None, None, "+", None],
            "lat_question": ["?", None, None, None, None],
            "lon_left": [123, None, 128, 354, None],
            "lon_right": [132, None, 158, 5, None],
            "lon_left_sign": [None, None, None, None, None],
            "lon_right_sign": [None, None, None, None, None],
            "lon_question": [None, None, "?", None, None],
            "first": [
                date(2000, 5, 5),
                date(2000, 5, 30),
                date(2000, 6, 1),
                date(2001, 1, 2),
                date(2000, 12, 30),
            ],
            "last": [
                date(2000, 5, 5),
                date(2000, 5, 31),
                date(2000, 6, 3),
                date(2001, 1, 7),
                date(2000, 12, 31),
            ],
            "over": [False, True, False, False, True],
        },
        schema={
            "ns": pl.Utf8,
            "no": pl.UInt32,
            "lat_left": pl.UInt8,
            "lat_right": pl.UInt8,
            "lat_left_sign": pl.Utf8,
            "lat_right_sign": pl.Utf8,
            "lat_question": pl.Utf8,
            "lon_left": pl.UInt16,
            "lon_right": pl.UInt16,
            "lon_left_sign": pl.Utf8,
            "lon_right_sign": pl.Utf8,
            "lon_question": pl.Utf8,
            "first": pl.Date,
            "last": pl.Date,
            "over": pl.Boolean,
        },
    )
    df_correct = pl.LazyFrame(
        {
            "ns": ["N", "N", "S"],
            "no": [1, 2, 1],
            "lat_left": [12, 12, 1],
            "lat_right": [12, 15, 5],
            "lat_left_sign": [None, None, "-"],
            "lat_right_sign": [None, None, "+"],
            "lat_question": ["?", None, None],
            "lon_left": [123, 128, 354],
            "lon_right": [132, 158, 5],
            "lon_left_sign": [None, None, None],
            "lon_right_sign": [None, None, None],
            "lon_question": [None, "?", None],
            "first": [date(2000, 5, 5), date(2000, 5, 30), date(2000, 12, 30)],
            "last": [date(2000, 5, 5), date(2000, 6, 3), date(2001, 1, 7)],
            "over": [False, False, False],
        },
        schema={
            "ns": pl.Utf8,
            "no": pl.UInt32,
            "lat_left": pl.UInt8,
            "lat_right": pl.UInt8,
            "lat_left_sign": pl.Utf8,
            "lat_right_sign": pl.Utf8,
            "lat_question": pl.Utf8,
            "lon_left": pl.UInt16,
            "lon_right": pl.UInt16,
            "lon_left_sign": pl.Utf8,
            "lon_right_sign": pl.Utf8,
            "lon_question": pl.Utf8,
            "first": pl.Date,
            "last": pl.Date,
            "over": pl.Boolean,
        },
    )
    df_out = ar_merge.merge(df_in)
    assert_frame_equal(df_out, df_correct, check_column_order=False)
