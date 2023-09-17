from datetime import date

import polars as pl
import pytest

import ar_old


@pytest.mark.parametrize(
    ("in_first", "in_last", "in_year", "in_month", "out_first", "out_last"),
    [
        (3, 3, 1970, 5, date(1970, 5, 3), date(1970, 5, 3)),
        (3, 5, 1970, 5, date(1970, 5, 3), date(1970, 5, 5)),
        (30, 31, 1970, 12, date(1970, 12, 30), date(1970, 12, 31)),
    ],
)
def test_calc_obs_date(
    in_first: int,
    in_last: int,
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
    )
    df_out = ar_old.calc_obs_date(df_in, in_year, in_month).collect()
    assert df_out.item(0, "first") == out_first
    assert df_out.item(0, "last") == out_last
