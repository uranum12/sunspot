from datetime import date

import polars as pl

import seiryo_sunspot_number_with_silso_plotly


def test_draw_sunspot_number_with_silso() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1)],
            "seiryo": [1, 2, 3],
            "silso": [2, 4, 6],
        },
        schema={"date": pl.Date, "seiryo": pl.Float64, "silso": pl.Float64},
    )
    _ = seiryo_sunspot_number_with_silso_plotly.draw_sunspot_number_with_silso_plotly(  # noqa: E501
        df
    )


def test_draw_scatter() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1)],
            "seiryo": [1, 2, 3],
            "silso": [2, 4, 6],
        },
        schema={"date": pl.Date, "seiryo": pl.Float64, "silso": pl.Float64},
    )
    factor = 0.5
    r2 = 1.0
    _ = seiryo_sunspot_number_with_silso_plotly.draw_scatter_plotly(
        df, factor, r2
    )


def test_draw_ratio_and_diff() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1)],
            "ratio": [0.4, 0.5, 0.45],
            "diff": [1, -2, 3],
        },
        schema={"date": pl.Date, "ratio": pl.Float64, "diff": pl.Float64},
    )
    _ = seiryo_sunspot_number_with_silso_plotly.draw_ratio_plotly(df)
    _ = seiryo_sunspot_number_with_silso_plotly.draw_diff_plotly(df)
    _ = seiryo_sunspot_number_with_silso_plotly.draw_ratio_and_diff_plotly(df)
