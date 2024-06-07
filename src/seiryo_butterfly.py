import json
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib.figure import Figure


@dataclass(frozen=True, slots=True, kw_only=True)
class DateDelta:
    """日付の間隔"""

    years: int = 0
    months: int = 0
    days: int = 0

    def __post_init__(self: "DateDelta") -> None:
        """初期化後の入力値チェック

        Raises:
            ValueError: 入力値が不正値の時に送出
        """
        values = self.to_dict().values()
        if all(value == 0 for value in values):
            msg = "all parameters cannot be zero"
            raise ValueError(msg)
        if any(value < 0 for value in values):
            msg = "parameters cannot be negative"
            raise ValueError(msg)

    def to_dict(self: "DateDelta") -> dict[str, int]:
        """辞書へ変換

        Returns:
            dict[str, int]: 変換後の辞書
        """
        return asdict(self)

    def _format_time(self: "DateDelta", time: int, unit: str) -> str:
        return f"{time}{unit}" if time != 0 else ""

    def to_interval(self: "DateDelta") -> str:
        """Polarsが受け取る期間の文字列へ変換

        Returns:
            str: 変換後の文字列
        """
        years = self._format_time(self.years, "y")
        months = self._format_time(self.months, "mo")
        days = self._format_time(self.days, "d")
        return f"{years}{months}{days}"

    def isoformat(self: "DateDelta") -> str:
        """ISO8601の形式へ変換

        Returns:
            str: 変換後の文字列
        """
        years = self._format_time(self.years, "Y")
        months = self._format_time(self.months, "M")
        days = self._format_time(self.days, "D")
        return f"P{years}{months}{days}"

    @classmethod
    def fromisoformat(cls: type["DateDelta"], data: str) -> "DateDelta":
        data = data.removeprefix("P")
        if "Y" in data:
            y_s, data = data.split("Y")
            y = int(y_s)
        else:
            y = 0
        if "M" in data:
            m_s, data = data.split("M")
            m = int(m_s)
        else:
            m = 0
        if "D" in data:
            d_s, data = data.split("D")
            d = int(d_s)
        else:
            d = 0
        return cls(years=y, months=m, days=d)


@dataclass(frozen=True, slots=True)
class ButterflyInfo:
    """蝶形図の情報"""

    lat_min: int
    lat_max: int
    date_start: date
    date_end: date
    date_interval: DateDelta

    def __post_init__(self: "ButterflyInfo") -> None:
        """初期化後の入力値チェック

        Raises:
            ValueError: 入力値が不正値の時に送出
        """
        if self.lat_min > self.lat_max:
            msg = "latitude minimum value cannot be greater than maximum value"
            raise ValueError(msg)
        if self.date_start > self.date_end:
            msg = "start date cannot be later than end date"
            raise ValueError(msg)

    def to_dict(self: "ButterflyInfo") -> dict[str, int | date | DateDelta]:
        """辞書へ変換

        Returns:
            dict[str, int | date | DateDelta]: 変換後の辞書
        """
        return {
            field.name: getattr(self, field.name) for field in fields(self)
        }

    def to_json(self: "ButterflyInfo") -> str:
        return json.dumps(
            self.to_dict(), default=lambda o: o.isoformat(), indent=2
        )

    @classmethod
    def from_dict(cls: type["ButterflyInfo"], data: dict) -> "ButterflyInfo":
        return cls(
            data["lat_min"],
            data["lat_max"],
            date.fromisoformat(data["date_start"]),
            date.fromisoformat(data["date_end"]),
            DateDelta.fromisoformat(data["date_interval"]),
        )


def calc_date_limit(df: pl.LazyFrame) -> tuple[date, date]:
    """日付の開始日と最終日を算出する

    Args:
        df (pl.LazyFrame): 黒点群データ

    Returns:
        tuple[date, date]: 開始日と最終日
    """
    date_range: dict[str, date] = (
        df.select(pl.min("date").alias("start"), pl.max("date").alias("end"))
        .collect()
        .to_dicts()[0]
    )
    return date_range["start"], date_range["end"]


def adjust_dates(start: date, end: date) -> tuple[date, date]:
    """日付の範囲を月初めに調整する

    Args:
        start (date): 開始日
        end (date): 最終日

    Returns:
        tuple[date, date]: 開始日と最終日
    """
    return start.replace(day=1), end.replace(day=1)


def agg_lat(df: pl.LazyFrame, interval: str) -> pl.LazyFrame:
    """緯度を期間ごとに集計.

    Args:
        df (pl.LazyFrame): 黒点群データ
        interval (str): 区切る期間

    Returns:
        pl.LazyFrame: 緯度データ
    """
    return (
        df.with_columns(pl.col("date").dt.truncate(interval))
        .drop_nulls()
        .group_by("date")
        .agg("lat_min", "lat_max")
        .rename({"lat_max": "max", "lat_min": "min"})
    )


def fill_lat(
    df: pl.LazyFrame, start: date, end: date, interval: str
) -> pl.LazyFrame:
    """範囲内の空白を空データで埋める

    Args:
        df (pl.LazyFrame): 緯度データ
        start (date): 開始日
        end (date): 終了日
        interval (str): 区切る期間

    Returns:
        pl.LazyFrame: 緯度データ
    """
    return (
        pl.LazyFrame({"date": pl.date_range(start, end, interval, eager=True)})
        .join(df, on="date", how="left", coalesce=True)
        .with_columns(pl.col("min", "max").fill_null(pl.lit([])))
        .sort("date")
    )


def calc_lat(df: pl.LazyFrame, info: ButterflyInfo) -> pl.DataFrame:
    """緯度データを算出する

    Args:
        df (pl.LazyFrame): 黒点群データ
        info (ButterflyInfo): 蝶形図の情報

    Returns:
        pl.DataFrame: 緯度データ
    """
    return (
        df.pipe(agg_lat, interval=info.date_interval.to_interval())
        .pipe(
            fill_lat,
            start=info.date_start,
            end=info.date_end,
            interval=info.date_interval.to_interval(),
        )
        .collect()
    )


def create_line(
    data_min: list[int], data_max: list[int], lat_min: int, lat_max: int
) -> npt.NDArray[np.uint8]:
    """蝶形図の一列分のデータを作成する

    Args:
        data_min (list[int]): データの最小値
        data_max (list[int]): データの最大値
        lat_min (int): 緯度の最小値
        lat_max (int): 緯度の最大値

    Returns:
        npt.NDArray[np.uint8]: 一列分のデータ
    """
    # numpy配列に変換
    arr_min = np.array(data_min)
    arr_max = np.array(data_max)

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
    for i_min, i_max in zip(index_min, index_max, strict=False):
        line[i_min:i_max] = 1

    return line


def create_image(
    df: pl.DataFrame, info: ButterflyInfo
) -> npt.NDArray[np.uint8]:
    """緯度データから蝶形図のデータを作成する

    Args:
        df (pl.DataFrame): 緯度データ
        info (ButterflyInfo): 蝶形図の情報

    Returns:
        npt.NDArray[np.uint8]: 蝶形図の画像データ
    """
    lines: list[npt.NDArray[np.uint8]] = []
    for data in df.iter_rows(named=True):
        data_min: list[int] = data["min"]
        data_max: list[int] = data["max"]
        line = create_line(data_min, data_max, info.lat_min, info.lat_max)
        lines.append(line.reshape(-1, 1))
    return np.hstack(lines)


def create_date_index(
    start: date, end: date, interval: str
) -> npt.NDArray[np.datetime64]:
    """蝶形図の日付のインデックスを作成する

    Args:
        start (date): 開始日
        end (date): 終了日
        interval (str): 期間

    Returns:
        npt.NDArray[np.datetime64]: 日付のインデックス
    """
    return pl.date_range(start, end, interval, eager=True).to_numpy()


def create_lat_index(lat_min: int, lat_max: int) -> npt.NDArray[np.int8]:
    """緯度のインデックスを作成する

    Args:
        lat_min (int): 緯度の最小値
        lat_max (int): 緯度の最大値

    Returns:
        npt.NDArray[np.int8]: 緯度のインデックス
    """
    lat_range = np.arange(lat_min, lat_max + 1, 1, dtype=np.int8)[::-1]
    return np.insert(np.abs(lat_range), np.arange(1, len(lat_range)), -1)


def draw_butterfly_diagram(
    img: npt.NDArray[np.uint8],
    info: ButterflyInfo,
    *,
    figsize: tuple[int, int] = (5, 8),
    year_interval: int = 10,
    lat_interval: int = 10,
) -> Figure:
    """蝶形図データを基に画像を作成する

    Args:
        img (npt.NDArray[np.uint8]): 蝶形図のデータ
        info (ButterflyInfo): 蝶形図の情報

    Returns:
        Figure: 作成した蝶形図
    """
    date_index = create_date_index(
        info.date_start, info.date_end, info.date_interval.to_interval()
    )
    lat_index = create_lat_index(info.lat_min, info.lat_max)

    xlabel = [
        (i, f"{d.year}")
        for i, d in enumerate(item.item() for item in date_index)
        if d.month == 1 and d.year % year_interval == 0
    ]
    ylabel = [(i, n) for i, n in enumerate(lat_index) if n % lat_interval == 0]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap="binary")

    ax.set_title("butterfly diagram")

    ax.set_xlabel("date")
    ax.set_xticks([i[0] for i in xlabel])
    ax.set_xticklabels([i[1] for i in xlabel])

    ax.set_ylabel("latitude")
    ax.set_yticks([i[0] for i in ylabel])
    ax.set_yticklabels([i[1] for i in ylabel])

    return fig


def main() -> None:
    data_path = Path("out/seiryo/all.parquet")
    output_path = Path("out/seiryo/butterfly")
    output_path.mkdir(parents=True, exist_ok=True)

    data_file = pl.scan_parquet(data_path)
    start, end = adjust_dates(*calc_date_limit(data_file))

    info = ButterflyInfo(-50, 50, start, end, DateDelta(months=1))
    pprint(info)

    df = calc_lat(data_file, info)
    df.write_parquet(output_path / "monthly.parquet")
    print(df)

    img = create_image(df, info)
    print(img)

    with (output_path / "monthly.npz").open("wb") as f_img:
        np.savez_compressed(f_img, img=img)

    with (output_path / "monthly.json").open("w") as f_info:
        f_info.write(info.to_json())

    fig_butterfly = draw_butterfly_diagram(img, info, year_interval=2)

    for f in ["png", "pdf"]:
        fig_butterfly.savefig(
            output_path / f"monthly.{f}",
            format=f,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )


if __name__ == "__main__":
    main()
