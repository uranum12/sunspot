from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import ar_new


@pytest.mark.parametrize(
    ("in_first", "in_last", "in_year", "in_month", "out_first", "out_last"),
    [
        ("apr.3", "apr.5", 2000, 4, date(2000, 4, 3), date(2000, 4, 5)),
        ("apr.30", "may.5", 2000, 4, date(2000, 4, 30), date(2000, 5, 5)),
        ("dec.30", "jan.5", 2000, 12, date(2000, 12, 30), date(2001, 1, 5)),
    ],
)
def test_calc_obs_date(
    in_first: str,
    in_last: str,
    in_year: int,
    in_month: int,
    out_first: date,
    out_last: date,
) -> None:
    df_in = pl.LazyFrame(
        {
            "first": [in_first],
            "last": [in_last],
        },
        schema={
            "first": pl.Utf8,
            "last": pl.Utf8,
        },
    )
    df_expected = pl.LazyFrame(
        {
            "first": [out_first],
            "last": [out_last],
        },
        schema={
            "first": pl.Date,
            "last": pl.Date,
        },
    )
    df_out = ar_new.calc_obs_date(df_in, in_year, in_month)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
    )
