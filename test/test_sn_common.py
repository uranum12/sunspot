from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import sn_common


@pytest.mark.parametrize(
    ("in_year", "in_month", "in_date", "out_date"),
    [
        (1956, 2, 18, date(1956, 2, 18)),
        (1972, 7, 30, date(1972, 7, 30)),
    ],
)
def test_calc_date(
    in_year: int,
    in_month: int,
    in_date: int,
    out_date: date,
) -> None:
    df_in = pl.LazyFrame(
        {
            "date": [in_date],
        },
        schema={
            "date": pl.UInt8,
        },
    )
    df_expected = pl.LazyFrame(
        {
            "date": [out_date],
        },
        schema={
            "date": pl.Date,
        },
    )
    df_out = sn_common.calc_date(df_in, in_year, in_month)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
    )


@pytest.mark.parametrize(
    ("in_date", "out_date"),
    [
        (
            [date(2000, 4, 2), date(2001, 6, 3), date(2000, 2, 3)],
            [date(2000, 2, 3), date(2000, 4, 2), date(2001, 6, 3)],
        ),
        (
            [date(1969, 12, 25), None, date(1970, 1, 1)],
            [None, date(1969, 12, 25), date(1970, 1, 1)],
        ),
    ],
)
def test_sort_by_date(in_date: list[date], out_date: list[date]) -> None:
    df_in = pl.LazyFrame(
        {
            "date": in_date,
        },
        schema={
            "date": pl.Date,
        },
    )
    df_expected = pl.LazyFrame(
        {
            "date": out_date,
        },
        schema={
            "date": pl.Date,
        },
    )
    df_out = sn_common.sort(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
    )
