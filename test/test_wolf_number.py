from datetime import date
from math import isclose

import polars as pl
import pytest

import wolf_number


@pytest.mark.parametrize(
    (
        "in_ng",
        "in_nf",
        "in_sg",
        "in_sf",
        "out_tg",
        "out_tf",
        "out_nr",
        "out_sr",
        "out_tr",
    ),
    [
        (1, 2, 4, 30, 5, 32, 12, 70, 82),
        (0, 0, 4, 30, 4, 30, 0, 70, 70),
        (10, 120, 20, 240, 30, 360, 220, 440, 660),
    ],
)
def test_calc_wolf_number(
    in_ng: int,
    in_nf: int,
    in_sg: int,
    in_sf: int,
    out_tg: int,
    out_tf: int,
    out_nr: int,
    out_sr: int,
    out_tr: int,
) -> None:
    df_in = pl.LazyFrame(
        {
            "ng": [in_ng],
            "nf": [in_nf],
            "sg": [in_sg],
            "sf": [in_sf],
        },
        schema={
            "ng": pl.UInt8,
            "nf": pl.UInt16,
            "sg": pl.UInt8,
            "sf": pl.UInt16,
        },
    )
    df_out = wolf_number.calc_wolf_number(df_in).collect()
    assert df_out.item(0, "tg") == out_tg
    assert df_out.item(0, "tf") == out_tf
    assert df_out.item(0, "nr") == out_nr
    assert df_out.item(0, "sr") == out_sr
    assert df_out.item(0, "tr") == out_tr


@pytest.mark.parametrize(
    ("in_date", "in_data", "out_date", "out_data"),
    [
        (
            [date(1961, 4, 30), date(1961, 5, 1), date(1961, 5, 2)],
            [12, 24, 8],
            [date(1961, 4, 1), date(1961, 5, 1)],
            [12, 16],
        ),
        (
            [date(2020, 9, 4), date(2020, 9, 5), date(2020, 9, 6)],
            [12, 54, 23],
            [date(2020, 9, 1)],
            [29.6666666666],
        ),
    ],
)
def test_agg_monthly(
    in_date: list[date],
    in_data: list[int],
    out_date: list[date],
    out_data: list[float],
) -> None:
    df_in = pl.LazyFrame(
        {
            "date": in_date,
            "data": in_data,
        },
        schema={
            "date": pl.Date,
            "data": pl.UInt16,
        },
    )
    df_out = wolf_number.agg_monthly(df_in).collect()
    print(df_out)
    assert sorted(df_out.get_column("date").to_list()) == sorted(out_date)
    assert all(
        isclose(x, y)
        for x, y in zip(
            sorted(df_out.get_column("data").to_list()),
            sorted(out_data),
            strict=True,
        )
    )
