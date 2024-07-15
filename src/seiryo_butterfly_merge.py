import json
from pathlib import Path
from pprint import pprint

import numpy as np
import numpy.typing as npt
import polars as pl

import seiryo_butterfly
import seiryo_butterfly_image
from seiryo_butterfly import ButterflyInfo
from seiryo_butterfly_config import Color


def merge_info(info_list: list[ButterflyInfo]) -> ButterflyInfo:
    if len({info.date_interval for info in info_list}) != 1:
        msg = "Date interval must be equal"
        raise ValueError(msg)
    lat_min = min(info.lat_min for info in info_list)
    lat_max = max(info.lat_max for info in info_list)
    date_start = min(info.date_start for info in info_list)
    date_end = max(info.date_end for info in info_list)
    return ButterflyInfo(
        lat_min, lat_max, date_start, date_end, info_list[0].date_interval
    )


def calc_lat_size(info: ButterflyInfo) -> int:
    return (info.lat_max - info.lat_min) * 2 + 1


def calc_date_size(info: ButterflyInfo) -> int:
    return pl.date_range(
        info.date_start,
        info.date_end,
        info.date_interval.to_interval(),
        eager=True,
    ).len()


def create_merged_image(
    dfl: list[pl.DataFrame], info: ButterflyInfo
) -> npt.NDArray[np.uint16]:
    lat_size = calc_lat_size(info)
    date_size = calc_date_size(info)
    img: npt.NDArray[np.uint16] = np.zeros(
        (lat_size, date_size), dtype=np.uint16
    )
    for i, df in enumerate(dfl):
        img = img + (
            seiryo_butterfly_image.create_image(
                seiryo_butterfly.fill_lat(
                    df.lazy(),
                    info.date_start,
                    info.date_end,
                    info.date_interval.to_interval(),
                ).collect(),
                info,
            )
            << i
        ).astype(np.uint16)
    return img


def create_color_image(
    img: npt.NDArray[np.uint16], cmap: list[Color]
) -> npt.NDArray[np.uint8]:
    img_merged = np.full((*img.shape, 3), 0xFF, dtype=np.uint8)
    for i, c in enumerate(cmap, 1):
        img_merged[img == i] = (c.red, c.green, c.blue)
    return img_merged


def main() -> None:
    monthly_data_path = Path("out/seiryo/butterfly/trimmed_monthly.parquet")
    monthly_info_path = monthly_data_path.with_suffix(".json")
    fromtext_data_path = Path("out/seiryo/butterfly/trimmed_fromtext.parquet")
    fromtext_info_path = fromtext_data_path.with_suffix(".json")
    cmap_path = Path("config/color/merged.json")
    output_path = Path("out/seiryo/butterfly")

    monthly_data = pl.read_parquet(monthly_data_path)

    with monthly_info_path.open("r") as f_monthly_info:
        monthly_info = ButterflyInfo.from_dict(json.load(f_monthly_info))

    fromtext_data = pl.read_parquet(fromtext_data_path)

    with fromtext_info_path.open("r") as f_fromtext_info:
        fromtext_info = ButterflyInfo.from_dict(json.load(f_fromtext_info))

    info = merge_info([fromtext_info, monthly_info])
    pprint(info)

    img = create_merged_image([fromtext_data, monthly_data], info)
    print(img)

    with cmap_path.open("r") as f_cmap:
        json_data = json.load(f_cmap)
        cmap = [Color(**item) for item in json_data]

    img_color = create_color_image(img, cmap)

    with (output_path / "merged.npz").open("wb") as f_img:
        np.savez_compressed(f_img, img=img)

    with (output_path / "merged.json").open("w") as f_info:
        f_info.write(info.to_json())

    with (output_path / "merged_color.npz").open("wb") as f_img_color:
        np.savez_compressed(f_img_color, img=img_color)


if __name__ == "__main__":
    main()
