from datetime import date, timedelta
from typing import Literal

import numpy as np
import numpy.typing as npt
from dateutil.relativedelta import relativedelta


def create_line(
    data: list[dict[str, int]],
    lat_n_max: int,
    lat_s_max: int,
) -> npt.NDArray[np.uint8]:
    line = np.zeros(2 * (lat_n_max + lat_s_max) + 1, dtype=np.uint8)
    for i in data:
        i_min = 2 * (lat_n_max + i["lat_left"])
        i_max = 2 * (lat_n_max + i["lat_right"]) + 1
        line[i_min:i_max] = 1
    return line


def create_date_index(
    start: date,
    end: date,
    unit: Literal["M", "D"],
) -> npt.NDArray[np.datetime64]:
    match unit:
        case "D":
            index = np.arange(
                start,
                end + timedelta(days=1),
                dtype="datetime64[D]",
            )
        case "M":
            index = np.arange(
                start,
                end + relativedelta(months=1),
                dtype="datetime64[M]",
            )
    return index


def create_lat_index(lat_n_max: int, lat_s_max: int) -> npt.NDArray[np.int8]:
    return np.insert(
        np.abs(np.arange(-lat_n_max, lat_s_max + 1, dtype=np.int8)),
        np.arange(1, lat_n_max + lat_s_max + 1),
        -1,
    )
