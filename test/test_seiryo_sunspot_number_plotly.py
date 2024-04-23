from datetime import date

import polars as pl

import seiryo_sunspot_number_plotly as seiryo_sunspot_number


def test_draw_sunspot_number_whole_disk() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1)],
            "total": [1, 2, 3],
        },
        schema={"date": pl.Date, "total": pl.Float64},
    )
    _ = seiryo_sunspot_number.draw_sunspot_number_whole_disk_plotly(df)


def test_draw_sunspot_number_hemispheric() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1)],
            "north": [1, 2, 3],
            "south": [2, 1, 3],
        },
        schema={"date": pl.Date, "north": pl.Float64, "south": pl.Float64},
    )
    _ = seiryo_sunspot_number.draw_sunspot_number_hemispheric_plotly(df)
