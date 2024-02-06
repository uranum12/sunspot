from datetime import date

import numpy as np
import numpy.typing as npt
import polars as pl

import butterfly_type


def create_line(
    data_min: list[int],
    data_max: list[int],
    lat_min: int,
    lat_max: int,
    lat_step: int = 1,
) -> npt.NDArray[np.uint8]:
    # numpy配列に変換
    arr_min = np.array(data_min)
    arr_max = np.array(data_max)

    # ステップごとの数値へ変換
    arr_min //= lat_step
    arr_max //= lat_step
    lat_min //= lat_step
    lat_max //= lat_step

    # 範囲外のインデックス
    index_outer = (arr_max < lat_min) | (lat_max < arr_min)

    # 範囲外のデータを除去
    arr_min = arr_min[~index_outer]
    arr_max = arr_max[~index_outer]

    # 緯度のインデックスの最大値
    max_index = 2 * (lat_max - lat_min)

    # 赤道のインデックス
    equator_index = 2 * lat_max

    # 行のインデックス
    index_min = equator_index - arr_max * 2
    index_max = equator_index - arr_min * 2 + 1

    # 範囲内に収まるよう調整
    index_min = np.clip(index_min, 0, max_index)
    index_max = np.clip(index_max, 1, max_index + 1)

    # 行を埋める
    line = np.zeros(max_index + 1, dtype=np.uint8)
    for i_min, i_max in zip(index_min, index_max, strict=True):
        line[i_min:i_max] = 1
    return line


def create_date_index(
    start: date, end: date, step: butterfly_type.DateDelta
) -> npt.NDArray[np.datetime64]:
    years = f"{step.years}y" if step.years is not None else ""
    months = f"{step.months}mo" if step.months is not None else ""
    days = f"{step.days}d" if step.days is not None else ""
    interval = f"{years}{months}{days}"
    return pl.date_range(start, end, interval, eager=True).to_numpy()


def create_date_index_daily(
    start: date, end: date
) -> npt.NDArray[np.datetime64]:
    return create_date_index(start, end, butterfly_type.DateDelta(days=1))


def create_date_index_monthly(
    start: date, end: date
) -> npt.NDArray[np.datetime64]:
    return create_date_index(
        start.replace(day=1), end, butterfly_type.DateDelta(months=1)
    )


def create_lat_index(
    lat_min: int, lat_max: int, lat_step: int = 1
) -> npt.NDArray[np.int8]:
    lat_range = np.arange(lat_min, lat_max + 1, lat_step, dtype=np.int8)[::-1]
    return np.insert(np.abs(lat_range), np.arange(1, len(lat_range)), -1)
