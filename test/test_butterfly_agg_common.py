from datetime import date

import numpy as np
import pytest

import butterfly_agg_common


@pytest.mark.parametrize(
    ("in_data_min", "in_data_max", "in_lat_min", "in_lat_max", "out_line"),
    [
        (
            [2, -3, 4],
            [3, -2, 4],
            -5,
            5,
            # 5    4     3     2     1     0     1     2     3     4     5
            [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        ),
        (
            [-5, -4, -1, 2, 3],
            [-4, -3, 1, 4, 4],
            -5,
            5,
            # 5    4     3     2     1     0     1     2     3     4     5
            [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        ),
        (
            [-8, -6, 0, 2, 3],
            [-7, -4, 0, 4, 10],
            -5,
            5,
            # 5    4     3     2     1     0     1     2     3     4     5
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ),
        (
            [-2, 1],
            [-1, 1],
            -2,
            3,
            # 3    2     1     0     1     2
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
        ),
        (
            [4],
            [4],
            2,
            5,
            # 5    4     3     2
            [0, 0, 1, 0, 0, 0, 0],
        ),
        (
            [4, -1],
            [1, -4],
            -1,
            1,
            # 1    0     1
            [1, 0, 0, 0, 1],
        ),
        (
            [4, -4],
            [4, -4],
            -1,
            1,
            # 1    0     1
            [0, 0, 0, 0, 0],
        ),
        (
            [],
            [],
            -1,
            1,
            # 1    0     1
            [0, 0, 0, 0, 0],
        ),
    ],
)
def test_create_line(
    in_data_min: list[int],
    in_data_max: list[int],
    in_lat_min: int,
    in_lat_max: int,
    out_line: list[int],
) -> None:
    out = butterfly_agg_common.create_line(
        in_data_min,
        in_data_max,
        in_lat_min,
        in_lat_max,
    )
    np.testing.assert_equal(out, out_line)


@pytest.mark.parametrize(
    ("in_start", "in_end", "out_index"),
    [
        (
            date(2020, 4, 4),
            date(2020, 4, 6),
            [
                np.datetime64("2020-04-04"),
                np.datetime64("2020-04-05"),
                np.datetime64("2020-04-06"),
            ],
        ),
        (
            date(2020, 2, 28),
            date(2020, 3, 1),
            [
                np.datetime64("2020-02-28"),
                np.datetime64("2020-02-29"),
                np.datetime64("2020-03-01"),
            ],
        ),
        (
            date(2021, 2, 28),
            date(2021, 3, 1),
            [
                np.datetime64("2021-02-28"),
                np.datetime64("2021-03-01"),
            ],
        ),
        (
            date(1969, 12, 30),
            date(1970, 1, 2),
            [
                np.datetime64("1969-12-30"),
                np.datetime64("1969-12-31"),
                np.datetime64("1970-01-01"),
                np.datetime64("1970-01-02"),
            ],
        ),
    ],
)
def test_create_date_index_dayly(
    in_start: date,
    in_end: date,
    out_index: list[date],
) -> None:
    out = butterfly_agg_common.create_date_index_daily(in_start, in_end)
    np.testing.assert_equal(out, out_index)


@pytest.mark.parametrize(
    ("in_start", "in_end", "out_index"),
    [
        (
            date(2020, 4, 4),
            date(2020, 7, 16),
            [
                np.datetime64("2020-04"),
                np.datetime64("2020-05"),
                np.datetime64("2020-06"),
                np.datetime64("2020-07"),
            ],
        ),
        (
            date(2020, 2, 28),
            date(2020, 3, 1),
            [
                np.datetime64("2020-02"),
                np.datetime64("2020-03"),
            ],
        ),
        (
            date(1969, 12, 30),
            date(1970, 1, 2),
            [
                np.datetime64("1969-12"),
                np.datetime64("1970-01"),
            ],
        ),
    ],
)
def test_create_date_index_monthly(
    in_start: date,
    in_end: date,
    out_index: list[date],
) -> None:
    out = butterfly_agg_common.create_date_index_monthly(in_start, in_end)
    np.testing.assert_equal(out, out_index)


@pytest.mark.parametrize(
    ("in_lat_n_max", "in_lat_s_max", "out_index"),
    [
        (-3, 2, [2, -1, 1, -1, 0, -1, 1, -1, 2, -1, 3]),  # N2 ~ S3
        (1, 4, [4, -1, 3, -1, 2, -1, 1]),  # N4 ~ N1
        (-3, -1, [1, -1, 2, -1, 3]),  # S1 ~ S3
    ],
)
def test_create_lat_index(
    in_lat_n_max: int,
    in_lat_s_max: int,
    out_index: list[int],
) -> None:
    out = butterfly_agg_common.create_lat_index(in_lat_n_max, in_lat_s_max)
    np.testing.assert_equal(out, out_index)
