from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import seiryo_check_data


@pytest.mark.parametrize(
    ("df_in", "df_expected"),
    [
        (
            (
                pl.DataFrame(
                    {
                        "col1": [1, 2, 3],
                        "col2": [-1, None, -3],
                        "col3": ["foo", "bar", None],
                    },
                    schema={
                        "col1": pl.UInt16,
                        "col2": pl.Int8,
                        "col3": pl.Utf8,
                    },
                )
            ),
            pl.DataFrame(
                {
                    "col1": [2, 3],
                    "col2": [None, -3],
                    "col3": ["bar", None],
                },
                schema={
                    "col1": pl.UInt16,
                    "col2": pl.Int8,
                    "col3": pl.Utf8,
                },
            ),
        ),
        (
            pl.DataFrame(
                {
                    "col1": [1, 2, 3],
                    "col2": [-1, -2, -3],
                    "col3": ["foo", "bar", "baz"],
                },
                schema={
                    "col1": pl.UInt16,
                    "col2": pl.Int8,
                    "col3": pl.Utf8,
                },
            ),
            pl.DataFrame(
                {
                    "col1": [],
                    "col2": [],
                    "col3": [],
                },
                schema={
                    "col1": pl.UInt16,
                    "col2": pl.Int8,
                    "col3": pl.Utf8,
                },
            ),
        ),
    ],
)
def test_find_null_values(
    df_in: pl.DataFrame,
    df_expected: pl.DataFrame,
) -> None:
    df_out = seiryo_check_data.find_null_values(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
    )


@pytest.mark.parametrize(
    ("in_date", "in_no", "out_date", "out_original", "out_expected"),
    [
        (
            [date(2020, 8, 20), date(2020, 8, 20), date(2020, 8, 20)],
            [1, 2, 1],
            [date(2020, 8, 20)],
            [[1, 1, 2]],
            [[1, 2, 3]],
        ),
        (
            [date(2020, 2, 2), date(2020, 2, 2), date(2020, 3, 3)],
            [1, 1, 2],
            [date(2020, 2, 2), date(2020, 3, 3)],
            [[1, 1], [2]],
            [[1, 2], [1]],
        ),
        (
            [date(2020, 12, 24), date(2020, 12, 24), date(2020, 12, 25)],
            [1, 2, 1],
            [],
            [],
            [],
        ),
    ],
)
def test_find_invalid_group_number(
    in_date: list[date],
    in_no: list[int],
    out_date: list[date],
    out_original: list[int],
    out_expected: list[int],
) -> None:
    df_in = pl.DataFrame(
        {
            "date": in_date,
            "no": in_no,
        },
        schema={
            "date": pl.Date,
            "no": pl.UInt8,
        },
    )
    df_expected = pl.DataFrame(
        {
            "date": out_date,
            "original": out_original,
            "expected": out_expected,
        },
        schema={
            "date": pl.Date,
            "original": pl.List(pl.UInt8),
            "expected": pl.List(pl.UInt8),
        },
    )
    df_out = seiryo_check_data.find_invalid_group_number(df_in).sort("date")
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
    )


@pytest.mark.parametrize(
    ("in_lat_min", "in_lat_max", "in_threshold", "out_lat_min", "out_lat_max"),
    [
        (
            [10, 20, 30, 40, 50],
            [-50, -40, -30, -20, -10],
            40,
            [10, 50],
            [-50, -10],
        ),
        (
            [1, 2, 3, 4, 5],
            [-5, -4, -3, -2, -1],
            10,
            [],
            [],
        ),
    ],
)
def test_find_invalid_lat_range(
    in_lat_min: list[int],
    in_lat_max: list[int],
    in_threshold: int,
    out_lat_min: list[int],
    out_lat_max: list[int],
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
    df_expected = pl.DataFrame(
        {
            "lat_min": out_lat_min,
            "lat_max": out_lat_max,
        },
        schema={
            "lat_min": pl.Int8,
            "lat_max": pl.Int8,
        },
    )
    df_out = seiryo_check_data.find_invalid_lat_range(df_in, in_threshold)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
        check_row_order=False,
    )


@pytest.mark.parametrize(
    (
        "in_lon_min",
        "in_lon_max",
        "in_min_threshold",
        "in_max_threshold",
        "out_lon_min",
        "out_lon_max",
    ),
    [
        (
            [10, 20, 30, 40, 50],
            [-50, -40, -30, -20, -10],
            -30,
            40,
            [10, 20, 50],
            [-50, -40, -10],
        ),
        (
            [1, 2, 3, 4, 5],
            [-5, -4, -3, -2, -1],
            -10,
            10,
            [],
            [],
        ),
    ],
)
def test_find_invalid_lon_range(
    in_lon_min: list[int],
    in_lon_max: list[int],
    in_min_threshold: int,
    in_max_threshold: int,
    out_lon_min: list[int],
    out_lon_max: list[int],
) -> None:
    df_in = pl.DataFrame(
        {
            "lon_min": in_lon_min,
            "lon_max": in_lon_max,
        },
        schema={
            "lon_min": pl.Int8,
            "lon_max": pl.Int8,
        },
    )
    df_expected = pl.DataFrame(
        {
            "lon_min": out_lon_min,
            "lon_max": out_lon_max,
        },
        schema={
            "lon_min": pl.Int8,
            "lon_max": pl.Int8,
        },
    )
    df_out = seiryo_check_data.find_invalid_lon_range(
        df_in,
        in_min_threshold,
        in_max_threshold,
    )
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
        check_row_order=False,
    )


@pytest.mark.parametrize(
    (
        "in_lat_min",
        "in_lat_max",
        "in_interval",
        "out_lat_min",
        "out_lat_max",
        "out_interval",
    ),
    [
        (
            [10, 10, 10, 10, 10],
            [10, 20, 30, 40, 50],
            20,
            [10, 10],
            [40, 50],
            [30, 40],
        ),
        (
            [1, 1, 1, 1, 1],
            [1, 2, 3, 4, 5],
            10,
            [],
            [],
            [],
        ),
    ],
)
def test_find_invalid_lat_interval(
    in_lat_min: list[int],
    in_lat_max: list[int],
    in_interval: int,
    out_lat_min: list[int],
    out_lat_max: list[int],
    out_interval: list[int],
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
    df_expected = pl.DataFrame(
        {
            "lat_min": out_lat_min,
            "lat_max": out_lat_max,
            "interval": out_interval,
        },
        schema={
            "lat_min": pl.Int8,
            "lat_max": pl.Int8,
            "interval": pl.Int8,
        },
    )
    df_out = seiryo_check_data.find_invalid_lat_interval(df_in, in_interval)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
        check_row_order=False,
    )


@pytest.mark.parametrize(
    (
        "in_lon_min",
        "in_lon_max",
        "in_interval",
        "out_lon_min",
        "out_lon_max",
        "out_interval",
    ),
    [
        (
            [10, 10, 10, 10, 10],
            [10, 20, 30, 40, 50],
            20,
            [10, 10],
            [40, 50],
            [30, 40],
        ),
        (
            [1, 1, 1, 1, 1],
            [1, 2, 3, 4, 5],
            10,
            [],
            [],
            [],
        ),
    ],
)
def test_find_invalid_lon_interval(
    in_lon_min: list[int],
    in_lon_max: list[int],
    in_interval: int,
    out_lon_min: list[int],
    out_lon_max: list[int],
    out_interval: list[int],
) -> None:
    df_in = pl.DataFrame(
        {
            "lon_min": in_lon_min,
            "lon_max": in_lon_max,
        },
        schema={
            "lon_min": pl.Int8,
            "lon_max": pl.Int8,
        },
    )
    df_expected = pl.DataFrame(
        {
            "lon_min": out_lon_min,
            "lon_max": out_lon_max,
            "interval": out_interval,
        },
        schema={
            "lon_min": pl.Int8,
            "lon_max": pl.Int8,
            "interval": pl.Int8,
        },
    )
    df_out = seiryo_check_data.find_invalid_lon_interval(df_in, in_interval)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
        check_row_order=False,
    )


@pytest.mark.parametrize(
    ("in_date", "out_date"),
    [
        (
            [
                date(2020, 2, 2),
                date(2020, 2, 2),
                date(2020, 3, 3),
                date(2020, 5, 5),
                date(2020, 5, 5),
            ],
            [
                date(2020, 2, 2),
                date(2020, 2, 2),
                date(2020, 5, 5),
                date(2020, 5, 5),
            ],
        ),
        (
            [
                date(2020, 1, 1),
                date(2020, 2, 2),
                date(2020, 3, 3),
                date(2020, 4, 4),
                date(2020, 5, 5),
            ],
            [],
        ),
    ],
)
def test_find_duplicate_date(
    in_date: list[date],
    out_date: list[date],
) -> None:
    df_in = pl.DataFrame(
        {
            "date": in_date,
        },
        schema={
            "date": pl.Date,
        },
    )
    df_expected = pl.DataFrame(
        {
            "date": out_date,
        },
        schema={
            "date": pl.Date,
        },
    )
    df_out = seiryo_check_data.find_duplicate_date(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
        check_row_order=False,
    )


@pytest.mark.parametrize(
    ("in_ng", "in_sg", "in_tg", "out_ng", "out_sg", "out_nsg", "out_tg"),
    [
        (
            [1, 2, 3, 4, 5],
            [2, 4, 6, 8, 10],
            [3, 7, 9, 12, 12],
            [2, 5],
            [4, 10],
            [6, 15],
            [7, 12],
        ),
        (
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [6, 6, 6, 6, 6],
            [],
            [],
            [],
            [],
        ),
    ],
)
def test_find_invalid_total_group(
    in_ng: list[int],
    in_sg: list[int],
    in_tg: list[int],
    out_ng: list[int],
    out_sg: list[int],
    out_nsg: list[int],
    out_tg: list[int],
) -> None:
    df_in = pl.DataFrame(
        {
            "ng": in_ng,
            "sg": in_sg,
            "tg": in_tg,
        },
        schema={
            "ng": pl.UInt8,
            "sg": pl.UInt8,
            "tg": pl.UInt8,
        },
    )
    df_expected = pl.DataFrame(
        {
            "ng": out_ng,
            "sg": out_sg,
            "nsg": out_nsg,
            "tg": out_tg,
        },
        schema={
            "ng": pl.UInt8,
            "sg": pl.UInt8,
            "nsg": pl.UInt8,
            "tg": pl.UInt8,
        },
    )
    df_out = seiryo_check_data.find_invalid_total_group(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
        check_row_order=False,
    )


@pytest.mark.parametrize(
    ("in_nf", "in_sf", "in_tf", "out_nf", "out_sf", "out_nsf", "out_tf"),
    [
        (
            [1, 2, 3, 4, 5],
            [2, 4, 6, 8, 10],
            [3, 7, 9, 12, 12],
            [2, 5],
            [4, 10],
            [6, 15],
            [7, 12],
        ),
        (
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [6, 6, 6, 6, 6],
            [],
            [],
            [],
            [],
        ),
    ],
)
def test_find_invalid_total_number(
    in_nf: list[int],
    in_sf: list[int],
    in_tf: list[int],
    out_nf: list[int],
    out_sf: list[int],
    out_nsf: list[int],
    out_tf: list[int],
) -> None:
    df_in = pl.DataFrame(
        {
            "nf": in_nf,
            "sf": in_sf,
            "tf": in_tf,
        },
        schema={
            "nf": pl.UInt8,
            "sf": pl.UInt8,
            "tf": pl.UInt8,
        },
    )
    df_expected = pl.DataFrame(
        {
            "nf": out_nf,
            "sf": out_sf,
            "nsf": out_nsf,
            "tf": out_tf,
        },
        schema={
            "nf": pl.UInt8,
            "sf": pl.UInt8,
            "nsf": pl.UInt8,
            "tf": pl.UInt8,
        },
    )
    df_out = seiryo_check_data.find_invalid_total_number(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
        check_row_order=False,
    )
