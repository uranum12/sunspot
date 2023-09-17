from datetime import date

import polars as pl
import pytest

import ar_notebook


@pytest.mark.parametrize(
    ("in_year", "in_month", "out_first"),
    [
        (1957, 4, date(1957, 4, 1)),
        (1960, 8, date(1960, 8, 1)),
        (1963, 12, date(1963, 12, 1)),
        (1964, 1, date(1964, 1, 1)),
    ],
)
def test_calc_obs_date(in_year: int, in_month: int, out_first: date) -> None:
    df_in = pl.LazyFrame()
    df_out = ar_notebook.calc_obs_date(df_in, in_year, in_month).collect()
    assert df_out.item(0, "first") == out_first


def test_fill_blanks() -> None:
    df_in = pl.LazyFrame()
    df_out = ar_notebook.fill_blanks(df_in).collect()
    assert df_out.item(0, "lat_left_sign") is None
    assert df_out.item(0, "lat_right_sign") is None
    assert df_out.item(0, "lat_question") is None
    assert df_out.item(0, "lon_left") is None
    assert df_out.item(0, "lon_right") is None
    assert df_out.item(0, "lon_left_sign") is None
    assert df_out.item(0, "lon_right_sign") is None
    assert df_out.item(0, "lon_question") is None
    assert df_out.item(0, "last") is None
    assert df_out.item(0, "over") is None


@pytest.mark.parametrize(
    ("in_no", "out_no"),
    [
        ("1234_56", "123456"),
        ("4321_2", "43212"),
    ],
)
def test_concat_no(in_no: str, out_no: str) -> None:
    df_in = pl.LazyFrame(
        {
            "no": [in_no],
        },
    )
    df_out = ar_notebook.concat_no(df_in).collect()
    assert df_out.item(0, "no") == out_no
