import polars as pl
import pytest

import sn_ut


@pytest.mark.parametrize(
    ("in_time", "out_time"),
    [
        ("3:00", "12:00"),
        ("2:15", "11:15"),
        ("0:30", "09:30"),
        ("-20", "08:40"),
        ("-45", "08:15"),
        ("m2:00", "07:00"),
        ("m1:15", "07:45"),
    ],
)
def test_calc_time(in_time: str, out_time: str) -> None:
    df_in = pl.LazyFrame(
        {
            "time": [in_time],
        },
    )
    df_out = sn_ut.calc_time(df_in).collect()
    assert df_out.item(0, "time") == out_time
