from datetime import time

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import sn_ut


@pytest.mark.parametrize(
    ("in_time", "out_time"),
    [
        ("3:00", time(12, 00)),
        ("2:15", time(11, 15)),
        ("0:30", time(9, 30)),
        ("-20", time(8, 40)),
        ("-45", time(8, 15)),
        ("m2:00", time(7, 00)),
        ("m1:15", time(7, 45)),
    ],
)
def test_calc_time(in_time: str, out_time: time) -> None:
    df_in = pl.LazyFrame({"time": [in_time]}, schema={"time": pl.Utf8})
    df_expected = pl.LazyFrame({"time": [out_time]}, schema={"time": pl.Time})
    df_out = sn_ut.calc_time(df_in)
    assert_frame_equal(df_out, df_expected, check_column_order=False)
