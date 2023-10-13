from datetime import date
from io import StringIO

import polars as pl
import pytest

import butterfly_text


@pytest.mark.parametrize(
    ("in_lat_min", "in_lat_max", "out_min", "out_max"),
    [
        (
            [12, 15, 23],
            [12, 15, 23],
            [12, 15, 23],
            [12, 15, 23],
        ),
        (
            [12, 15, 15, 23, 23],
            [12, 15, 15, 23, 23],
            [12, 15, 23],
            [12, 15, 23],
        ),
        (
            [12, 14, 23, 24, 30, 31],
            [15, 18, 25, 25, 35, 32],
            [12, 23, 30],
            [18, 25, 35],
        ),
        (
            [12, 31, 14, 24, 23, 30],
            [15, 32, 18, 25, 25, 35],
            [12, 23, 30],
            [18, 25, 35],
        ),
        (
            [12, 13, 13, 23, 25, 26],
            [13, 13, 15, 25, 26, 27],
            [12, 13, 13, 23, 25, 26],
            [13, 13, 15, 25, 26, 27],
        ),
    ],
)
def test_merge_data(
    in_lat_min: list[int],
    in_lat_max: list[int],
    out_min: list[int],
    out_max: list[int],
) -> None:
    df_in = pl.DataFrame(
        {
            "lat_min": in_lat_min,
            "lat_max": in_lat_max,
        },
        schema={
            "lat_min": pl.Int8,
            "lat_max": pl.Int8,
        },
    )
    df_out = butterfly_text.merge_data(df_in)
    assert df_out.get_column("min").to_list() == out_min
    assert df_out.get_column("max").to_list() == out_max


@pytest.mark.parametrize(
    ("in_min", "in_max", "out_n_min", "out_n_max", "out_s_min", "out_s_max"),
    [
        (
            [-10, -6, 4, 8],
            [-8, -5, 6, 11],
            [4, 8],
            [6, 11],
            [5, 8],
            [6, 10],
        ),
        (
            [-11, -6, 10],
            [-11, 5, 10],
            [0, 10],
            [5, 10],
            [0, 11],
            [6, 11],
        ),
        (
            [0, 5],
            [2, 7],
            [0, 5],
            [2, 7],
            [0],
            [0],
        ),
        (
            [-10],
            [0],
            [0],
            [0],
            [0],
            [10],
        ),
        (
            [3],
            [10],
            [3],
            [10],
            [],
            [],
        ),
        (
            [-10],
            [-3],
            [],
            [],
            [3],
            [10],
        ),
    ],
)
def test_split_data(
    in_min: list[int],
    in_max: list[int],
    out_n_min: list[int],
    out_n_max: list[int],
    out_s_min: list[int],
    out_s_max: list[int],
) -> None:
    df_in = pl.DataFrame(
        {
            "max": in_max,
            "min": in_min,
        },
        schema={
            "max": pl.Int8,
            "min": pl.Int8,
        },
    )
    df_out_n, df_out_s = butterfly_text.split_data(df_in)
    assert df_out_n.get_column("min").to_list() == out_n_min
    assert df_out_n.get_column("max").to_list() == out_n_max
    assert df_out_s.get_column("min").to_list() == out_s_min
    assert df_out_s.get_column("max").to_list() == out_s_max


@pytest.mark.parametrize(
    ("in_min", "in_max", "out_str"),
    [
        (
            [12, 18, 23],
            [15, 20, 23],
            "12-15 18-20 23-23",
        ),
        (
            [12],
            [15],
            "12-15",
        ),
        (
            [],
            [],
            "",
        ),
    ],
)
def test_convert_to_str(
    in_min: list[str],
    in_max: list[int],
    out_str: str,
) -> None:
    df_in = pl.DataFrame(
        {
            "min": in_min,
            "max": in_max,
        },
        schema={
            "min": pl.Int8,
            "max": pl.Int8,
        },
    )
    assert butterfly_text.convert_to_str(df_in) == out_str


@pytest.mark.parametrize(
    ("in_start", "in_end", "out_file"),
    [
        (
            date(1960, 1, 1),
            date(1965, 5, 1),
            "//Data File for Butterfly Diagram\n"
            ">>1960/01-1965/05\n\n"
            "<----data---->\n",
        ),
        (
            date(2000, 1, 1),
            date(2010, 12, 1),
            "//Data File for Butterfly Diagram\n"
            ">>2000/01-2010/12\n\n"
            "<----data---->\n",
        ),
    ],
)
def test_write_header(in_start: date, in_end: date, out_file: str) -> None:
    vfile = StringIO()
    butterfly_text.write_header(vfile, in_start, in_end)
    assert vfile.getvalue() == out_file


@pytest.mark.parametrize(
    ("in_date", "in_data_n", "in_data_s", "out_file"),
    [
        (
            date(1963, 4, 1),
            "12-15 23-27",
            "2-3",
            "1963/04/N:12-15 23-27\n1963/04/S:2-3\n",
        ),
        (
            date(2010, 11, 1),
            "0-0 1-2 3-4",
            "",
            "2010/11/N:0-0 1-2 3-4\n2010/11/S:\n",
        ),
        (
            date(2000, 1, 1),
            "",
            "",
            "2000/01/N:\n2000/01/S:\n",
        ),
    ],
)
def test_write_data(
    in_date: date,
    in_data_n: str,
    in_data_s: str,
    out_file: str,
) -> None:
    vfile = StringIO()
    butterfly_text.write_data(vfile, in_date, in_data_n, in_data_s)
    assert vfile.getvalue() == out_file
