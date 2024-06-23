import json
from datetime import date
from pathlib import Path
from pprint import pprint

import numpy as np
import numpy.typing as npt
import polars as pl

import seiryo_butterfly
from seiryo_butterfly_config import ButterflyDiagram


def create_merged_image(
    dfl: list[pl.DataFrame], info: seiryo_butterfly.ButterflyInfo
) -> npt.NDArray[np.uint16]:
    lat_size = len(
        seiryo_butterfly.create_lat_index(info.lat_min, info.lat_max)
    )
    date_size = len(
        seiryo_butterfly.create_date_index(
            info.date_start, info.date_end, info.date_interval.to_interval()
        )
    )
    img: npt.NDArray[np.uint16] = np.zeros(
        (lat_size, date_size), dtype=np.uint16
    )
    for i, df in enumerate(dfl):
        img = img + (
            seiryo_butterfly.create_image(
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
    img: npt.NDArray[np.uint16], cmap: list[tuple[int, int, int]]
) -> npt.NDArray[np.uint8]:
    img_merged = np.full((*img.shape, 3), 0xFF, dtype=np.uint8)
    for i, c in enumerate(cmap, 1):
        img_merged[img == i] = c
    return img_merged


def main() -> None:
    config_path = Path("config/seiryo/butterfly_diagram")
    output_path = Path("out/seiryo/butterfly")

    df1 = pl.read_parquet(Path("out/seiryo/butterfly/fromtext.parquet"))
    df2 = pl.read_parquet(Path("out/seiryo/butterfly/monthly.parquet"))
    df2_date_min = df2.select(pl.min("date")).item()
    dfl = [df1.filter(pl.col("date") < df2_date_min), df2]

    info = seiryo_butterfly.ButterflyInfo(
        -50,
        50,
        date(1950, 1, 1),
        date(2024, 12, 1),
        seiryo_butterfly.DateDelta(months=1),
    )
    pprint(info)

    img = create_merged_image(dfl, info)
    print(img)

    with (output_path / "merged.npz").open("wb") as f_img:
        np.savez_compressed(f_img, img=img)

    with (output_path / "merged.json").open("w") as f_info:
        f_info.write(info.to_json())

    cmap = [(0x00, 0x00, 0x00), (0xFF, 0x00, 0x00), (0xFF, 0x00, 0x00)]
    img_color = create_color_image(img, cmap)

    with (config_path / "merged.json").open("r") as file:
        config = ButterflyDiagram(**json.load(file))

    fig = seiryo_butterfly.draw_butterfly_diagram(img_color, info, config)

    for f in ["png", "pdf"]:
        fig.savefig(
            output_path / f"merged.{f}",
            format=f,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )


if __name__ == "__main__":
    main()
