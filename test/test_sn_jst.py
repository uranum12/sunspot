from datetime import time

import polars as pl
import pytest

import sn_jst


@pytest.mark.parametrize(
    ("in_time", "out_time"),
    [
        ("12:30", time(12, 30)),
        ("9:30", time(9, 30)),
        ("15:00", time(15, 0)),
    ],
)
def test_calc_time(in_time: str, out_time: time) -> None:
    df_in = pl.LazyFrame(
        {
            "time": [in_time],
        },
    )
    df_out = sn_jst.calc_time(df_in).collect()
    assert df_out.item(0, "time") == out_time
