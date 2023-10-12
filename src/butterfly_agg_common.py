from datetime import date, timedelta

import numpy as np
import numpy.typing as npt
from dateutil.relativedelta import relativedelta


def create_line(
    data_min: list[int],
    data_max: list[int],
    lat_min: int,
    lat_max: int,
) -> npt.NDArray[np.uint8]:
    line_size = 2 * (lat_max - lat_min) + 1
    line = np.zeros(line_size, dtype=np.uint8)
    for i_min, i_max in zip(
        np.clip(2 * (lat_max + np.array(data_min)), 0, line_size),
        np.clip(2 * (lat_max + np.array(data_max)) + 1, 0, line_size),
        strict=False,
    ):
        line[i_min:i_max] = 1
    return line


def create_date_index_daily(
    start: date,
    end: date,
) -> npt.NDArray[np.datetime64]:
    return np.arange(
        start,
        end + timedelta(days=1),
        dtype="datetime64[D]",
    )


def create_date_index_monthly(
    start: date,
    end: date,
) -> npt.NDArray[np.datetime64]:
    return np.arange(
        start,
        end + relativedelta(months=1),
        dtype="datetime64[M]",
    )


def create_lat_index(lat_min: int, lat_max: int) -> npt.NDArray[np.int8]:
    lat_range = np.arange(lat_min, lat_max + 1, dtype=np.int8)[::-1]
    return np.insert(np.abs(lat_range), np.arange(1, len(lat_range)), -1)
