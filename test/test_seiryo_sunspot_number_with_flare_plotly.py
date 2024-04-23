from datetime import date

import polars as pl

import seiryo_sunspot_number_with_flare_plotly


def test_draw_sunspot_number_with_flare() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1)],
            "seiryo": [1, 2, 3],
            "flare": [1, 2, 3],
        },
        schema={"date": pl.Date, "seiryo": pl.Float64, "flare": pl.Float64},
    )
    _ = seiryo_sunspot_number_with_flare_plotly.draw_sunspot_number_with_flare_plotly(  # noqa: E501
        df
    )
