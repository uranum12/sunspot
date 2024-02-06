from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

import butterfly_agg_common


def main() -> None:
    data_file = Path("data/seiryo/1950-2023.txt")
    output_path = Path("out/butterfly")
    output_path.mkdir(parents=True, exist_ok=True)

    with data_file.open() as f:
        df_file = (
            pl.LazyFrame({"txt": f.readlines()[4:]})
            .select(
                pl.col("txt").str.extract(r"^(\d+/\d+)/").alias("date"),
                pl.col("txt").str.extract(r"([NS])").alias("ns"),
                pl.col("txt").str.extract_all(r"(\d+-\d+)").alias("data"),
            )
            .with_columns(
                pl.when(pl.col("ns").eq("N"))
                .then(
                    pl.col("data").map_elements(
                        lambda a: [[int(a) for a in a.split("-")] for a in a]
                    )
                )
                .otherwise(
                    pl.col("data").map_elements(
                        lambda a: [
                            [-int(a) for a in a.split("-")[::-1]] for a in a
                        ]
                    )
                )
            )
            .explode("data")
            .select(
                pl.col("date").str.strptime(pl.Date, "%Y/%m"),
                pl.col("data").list.get(0).alias("min"),
                pl.col("data").list.get(1).alias("max"),
            )
            .drop_nulls()
            .collect()
        )

    start = df_file.select("date").min().item()
    end = df_file.select("date").max().item()

    lat_max = 50
    lat_min = -50

    data: list[npt.NDArray[np.uint8]] = []
    date_index = butterfly_agg_common.create_date_index_monthly(start, end)
    lat_index = butterfly_agg_common.create_lat_index(lat_min, lat_max)

    for i in date_index:
        df = (
            df_file.lazy()
            .filter(pl.col("date").eq(i.astype(date)))
            .drop("date")
            .collect()
        )

        df_data = df.to_dict(as_series=False)
        line = butterfly_agg_common.create_line(
            df_data["min"], df_data["max"], lat_min, lat_max
        )
        data.append(line.reshape(-1, 1))

    img = np.hstack(data)

    print(img)
    print(date_index)
    print(lat_index)

    with (output_path / "seiryo.npz").open("wb") as f:
        np.savez_compressed(f, img=img, date=date_index, lat=lat_index)


if __name__ == "__main__":
    main()
