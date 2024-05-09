from datetime import date

import polars as pl

import seiryo_obs_days_plotly


def test_draw_monthly_obs_days() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
            "obs": [3, 0, 2],
        },
        schema={"date": pl.Date, "obs": pl.UInt8},
    )
    _ = seiryo_obs_days_plotly.draw_monthly_obs_days_plotly(df)
