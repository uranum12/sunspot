import json
from datetime import date
from pathlib import Path
from pprint import pprint
from re import match

import numpy as np
import polars as pl

import seiryo_butterfly
from seiryo_butterfly_config import ButterflyDiagram


def load_txt_data(path: Path) -> tuple[date, date, list[str]]:
    with path.open("r") as f:
        lines = f.read().splitlines()
    if m := match(r">>(\d+)/(\d+)-(\d+)/(\d+)", lines[1]):
        start = date(int(m.group(1)), int(m.group(2)), 1)
        end = date(int(m.group(3)), int(m.group(4)), 1)
    else:
        msg = "invalid date range data"
        raise ValueError(msg)
    return start, end, lines[4:]


def extract_lat(txt: list[str]) -> pl.LazyFrame:
    pat = (
        r"(?P<year>\d+)/(?P<month>\d+)/(?<ns>[NS]):"
        r"(?P<lat>\d+-\d+(?: \d+-\d+)*)?"
    )
    return (
        pl.LazyFrame({"txt": txt})
        .select(pl.col("txt").str.extract_groups(pat))
        .unnest("txt")
        .with_columns(
            pl.col("lat")
            .str.split(by=" ")
            .list.eval(pl.element().str.split(by="-"))
        )
        .explode("lat")
        .drop_nulls()
        .cast({"lat": pl.List(pl.Int8)})
        .with_columns(
            pl.when(pl.col("ns").eq("N"))
            .then(pl.col("lat"))
            .otherwise(pl.col("lat").list.eval(-pl.element()))
        )
        .select(
            pl.date("year", "month", 1).alias("date"),
            pl.col("lat").list.min().alias("lat_min"),
            pl.col("lat").list.max().alias("lat_max"),
        )
    )


def main() -> None:
    data_path = Path("data/seiryo/1950-2023.txt")
    config_path = Path("config/seiryo/butterfly_diagram")
    output_path = Path("out/seiryo/butterfly")
    output_path.mkdir(parents=True, exist_ok=True)

    start, end, txt = load_txt_data(data_path)
    lf = extract_lat(txt)

    info = seiryo_butterfly.ButterflyInfo(
        -50, 50, start, end, seiryo_butterfly.DateDelta(months=1)
    )
    pprint(info)

    df = seiryo_butterfly.calc_lat(lf, info)
    df.write_parquet(output_path / "fromtext.parquet")
    print(df)

    img = seiryo_butterfly.create_image(df, info)
    print(img)

    with (output_path / "fromtext.npz").open("wb") as f_img:
        np.savez_compressed(f_img, img=img)

    with (output_path / "fromtext.json").open("w") as f_info:
        f_info.write(info.to_json())

    with (config_path / "fromtext.json").open("r") as file:
        config = ButterflyDiagram(**json.load(file))

    fig_butterfly = seiryo_butterfly.draw_butterfly_diagram(img, info, config)

    for f in ["png", "pdf"]:
        fig_butterfly.savefig(
            output_path / f"fromtext.{f}",
            format=f,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )


if __name__ == "__main__":
    main()
