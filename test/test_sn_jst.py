import polars as pl
import pytest

import sn_jst


@pytest.mark.parametrize(
    ("in_time", "out_time"),
    [
        ("12:30", "12:30"),
        ("9:30", "09:30"),
        ("15:00", "15:00"),
    ],
)
def test_calc_time(in_time: str, out_time: str) -> None:
    df_in = pl.LazyFrame(
        {
            "time": [in_time],
        },
    )
    df_out = sn_jst.calc_time(df_in).collect()
    assert df_out.item(0, "time") == out_time
