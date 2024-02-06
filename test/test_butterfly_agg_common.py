from datetime import date

import numpy as np
import pytest

import butterfly_agg_common
import butterfly_type


@pytest.mark.parametrize(
    (
        "in_data_min",
        "in_data_max",
        "in_lat_min",
        "in_lat_max",
        "in_lat_step",
        "out_line",
    ),
    [
        (
            [2, -3, 4],
            [3, -2, 4],
            -5,
            5,
            None,
            # 5    4     3     2     1     0     1     2     3     4     5
            [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        ),
        (
            [-5, -4, -1, 2, 3],
            [-4, -3, 1, 4, 4],
            -5,
            5,
            None,
            # 5    4     3     2     1     0     1     2     3     4     5
            [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        ),
        (
            [-8, -6, 0, 2, 3],
            [-7, -4, 0, 4, 10],
            -5,
            5,
            None,
            # 5    4     3     2     1     0     1     2     3     4     5
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ),
        (
            [-2, 1],
            [-1, 1],
            -2,
            3,
            None,
            # 3    2     1     0     1     2
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
        ),
        (
            [4],
            [4],
            2,
            5,
            None,
            # 5    4     3     2
            [0, 0, 1, 0, 0, 0, 0],
        ),
        (
            [1, -4],
            [4, -1],
            -1,
            1,
            None,
            # 1    0     1
            [1, 0, 0, 0, 1],
        ),
        (
            [4, -4],
            [4, -4],
            -1,
            1,
            None,
            # 1    0     1
            [0, 0, 0, 0, 0],
        ),
        (
            [],
            [],
            -1,
            1,
            None,
            # 1    0     1
            [0, 0, 0, 0, 0],
        ),
        (
            [-8, -2, 2, 4],
            [-6, -2, 4, 8],
            -6,
            6,
            2,
            # 6    4     2     0     2     4     6
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        ),
        (
            [-2, 2],
            [-2, 6],
            -6,
            6,
            4,
            # 6    2     2     6
            [1, 1, 1, 0, 1, 0, 0],
        ),
        (
            # 中途半端な数値は通常S方向に引っ張られる
            [-3, 1, 5, 8],
            [-1, 1, 5, 8],
            -6,
            6,
            2,
            # 6    4     2     0     2     4     6
            [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
        ),
        (
            # 赤道の値が存在しない範囲だと
            # 中途半端な数値は通常N方向に引っ張られる
            [-6, -2, 0],
            [-4, -2, 2],
            -5,
            5,
            2,
            # 5    3     1     1     3     5
            [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
        ),
    ],
)
def test_create_line(
    in_data_min: list[int],
    in_data_max: list[int],
    in_lat_min: int,
    in_lat_max: int,
    in_lat_step: int | None,
    out_line: list[int],
) -> None:
    if in_lat_step is None:
        out = butterfly_agg_common.create_line(
            in_data_min, in_data_max, in_lat_min, in_lat_max
        )
    else:
        out = butterfly_agg_common.create_line(
            in_data_min, in_data_max, in_lat_min, in_lat_max, in_lat_step
        )
    np.testing.assert_equal(out, out_line)


@pytest.mark.parametrize(
    ("in_start", "in_end", "in_step", "out_index"),
    [
        (
            date(2020, 4, 4),
            date(2020, 4, 6),
            butterfly_type.DateDelta(days=1),
            [
                np.datetime64("2020-04-04"),
                np.datetime64("2020-04-05"),
                np.datetime64("2020-04-06"),
            ],
        ),
        (
            date(2020, 2, 28),
            date(2020, 3, 5),
            butterfly_type.DateDelta(days=2),
            [
                np.datetime64("2020-02-28"),
                np.datetime64("2020-03-01"),
                np.datetime64("2020-03-03"),
                np.datetime64("2020-03-05"),
            ],
        ),
        (
            date(2021, 2, 28),
            date(2021, 3, 4),
            butterfly_type.DateDelta(days=3),
            [np.datetime64("2021-02-28"), np.datetime64("2021-03-03")],
        ),
        (
            date(2020, 4, 4),
            date(2020, 7, 16),
            butterfly_type.DateDelta(months=2),
            [np.datetime64("2020-04-04"), np.datetime64("2020-06-04")],
        ),
        (
            date(1969, 11, 1),
            date(1970, 2, 2),
            butterfly_type.DateDelta(months=1),
            [
                np.datetime64("1969-11-01"),
                np.datetime64("1969-12-01"),
                np.datetime64("1970-01-01"),
                np.datetime64("1970-02-01"),
            ],
        ),
        (
            date(2020, 2, 28),
            date(2020, 4, 1),
            butterfly_type.DateDelta(months=1, days=1),
            [np.datetime64("2020-02-28"), np.datetime64("2020-03-29")],
        ),
        (
            date(2020, 4, 1),
            date(2023, 12, 4),
            butterfly_type.DateDelta(years=1, months=2, days=3),
            [
                np.datetime64("2020-04-01"),
                np.datetime64("2021-06-04"),
                np.datetime64("2022-08-07"),
                np.datetime64("2023-10-10"),
            ],
        ),
    ],
)
def test_create_date_index(
    in_start: date,
    in_end: date,
    in_step: butterfly_type.DateDelta,
    out_index: list[date],
) -> None:
    out = butterfly_agg_common.create_date_index(in_start, in_end, in_step)
    np.testing.assert_equal(out, out_index)


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
            [np.datetime64("2021-02-28"), np.datetime64("2021-03-01")],
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
    in_start: date, in_end: date, out_index: list[date]
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
            [np.datetime64("2020-02"), np.datetime64("2020-03")],
        ),
        (
            date(1969, 12, 30),
            date(1970, 1, 2),
            [np.datetime64("1969-12"), np.datetime64("1970-01")],
        ),
    ],
)
def test_create_date_index_monthly(
    in_start: date, in_end: date, out_index: list[date]
) -> None:
    out = butterfly_agg_common.create_date_index_monthly(in_start, in_end)
    np.testing.assert_equal(out, out_index)


@pytest.mark.parametrize(
    ("in_lat_min", "in_lat_max", "in_lat_step", "out_index"),
    [
        (-3, 2, None, [2, -1, 1, -1, 0, -1, 1, -1, 2, -1, 3]),  # S3 ~ N2
        (1, 4, None, [4, -1, 3, -1, 2, -1, 1]),  # N1 ~ N4
        (-3, -1, None, [1, -1, 2, -1, 3]),  # S3 ~ S1
        (-2, 2, 2, [2, -1, 0, -1, 2]),  # S2 ~ N2
        (0, 10, 5, [10, -1, 5, -1, 0]),  # 0 ~ N10
        (-4, -1, 2, [2, -1, 4]),  # S4 ~ S1
        (1, 4, 2, [3, -1, 1]),  # N1 ~ N4
    ],
)
def test_create_lat_index(
    in_lat_min: int,
    in_lat_max: int,
    in_lat_step: int | None,
    out_index: list[int],
) -> None:
    if in_lat_step is None:
        out = butterfly_agg_common.create_lat_index(in_lat_min, in_lat_max)
    else:
        out = butterfly_agg_common.create_lat_index(
            in_lat_min, in_lat_max, in_lat_step
        )
    np.testing.assert_equal(out, out_index)
