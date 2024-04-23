from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import seiryo_sunspot_number


@pytest.mark.parametrize(
    (
        "in_date",
        "in_ng",
        "in_nf",
        "in_sg",
        "in_sf",
        "in_tg",
        "in_tf",
        "out_date",
        "out_north",
        "out_south",
        "out_total",
    ),
    [
        pytest.param(
            [
                date(2020, 2, 1),
                date(2020, 2, 2),
                date(2020, 2, 3),
                date(2020, 2, 4),
                date(2020, 2, 5),
            ],
            [1, 2, 3, 4, 5],
            [2, 4, 6, 8, 10],
            [3, 2, 1, 0, 0],
            [9, 6, 3, 0, 0],
            [4, 4, 4, 4, 5],
            [11, 10, 9, 8, 10],
            [date(2020, 2, 1)],
            [36.0],
            [15.6],
            [51.6],
        ),
        pytest.param(
            [
                date(2020, 1, 30),
                date(2020, 1, 31),
                date(2020, 2, 1),
                date(2020, 2, 2),
                date(2020, 2, 3),
            ],
            [1, 2, 3, 4, 5],
            [2, 4, 6, 8, 10],
            [3, 2, 1, 0, 0],
            [9, 6, 3, 0, 0],
            [4, 4, 4, 4, 4],
            [11, 10, 9, 8, 7],
            [date(2020, 1, 1), date(2020, 2, 1)],
            [18.0, 48.0],
            [32.5, 4.33333],
            [50.5, 48.0],
        ),
    ],
)
def test_calc_sunspot_number(
    in_date: list[date],
    in_ng: list[int],
    in_nf: list[int],
    in_sg: list[int],
    in_sf: list[int],
    in_tg: list[int],
    in_tf: list[int],
    out_date: list[date],
    out_north: list[float],
    out_south: list[float],
    out_total: list[float],
) -> None:
    df_in = pl.DataFrame(
        {
            "date": in_date,
            "ng": in_ng,
            "nf": in_nf,
            "sg": in_sg,
            "sf": in_sf,
            "tg": in_tg,
            "tf": in_tf,
        },
        schema={
            "date": pl.Date,
            "ng": pl.UInt8,
            "nf": pl.UInt16,
            "sg": pl.UInt8,
            "sf": pl.UInt16,
            "tg": pl.UInt8,
            "tf": pl.UInt16,
        },
    )
    df_expected = pl.DataFrame(
        {
            "date": out_date,
            "north": out_north,
            "south": out_south,
            "total": out_total,
        },
        schema={
            "date": pl.Date,
            "north": pl.Float64,
            "south": pl.Float64,
            "total": pl.Float64,
        },
    )
    df_out = seiryo_sunspot_number.calc_sunspot_number(df_in)
    assert_frame_equal(df_out, df_expected)


def test_draw_sunspot_number_whole_disk() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1)],
            "total": [1, 2, 3],
        },
        schema={"date": pl.Date, "total": pl.Float64},
    )
    _ = seiryo_sunspot_number.draw_sunspot_number_whole_disk(df)


def test_draw_sunspot_number_hemispheric() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1)],
            "north": [1, 2, 3],
            "south": [2, 1, 3],
        },
        schema={"date": pl.Date, "north": pl.Float64, "south": pl.Float64},
    )
    _ = seiryo_sunspot_number.draw_sunspot_number_hemispheric(df)
