from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from dateutil.relativedelta import relativedelta


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
                        lambda a: [[int(a) for a in a.split("-")] for a in a],
                    ),
                )
                .otherwise(
                    pl.col("data").map_elements(
                        lambda a: [
                            [-int(a) for a in a.split("-")[::-1]] for a in a
                        ],
                    ),
                ),
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

    data: list[npt.NDArray[np.uint8]] = []
    index = np.arange(
        start,
        end + relativedelta(months=1),
        dtype="datetime64[M]",
    )

    for current in (i.astype(date) for i in index):
        df = (
            df_file.lazy()
            .filter(pl.col("date").eq(current))
            .drop("date")
            .collect()
        )

        line = np.zeros(201, dtype=np.uint8)
        for i in df.iter_rows(named=True):
            i_min = 100 + 2 * i["min"]
            i_max = 100 + 2 * i["max"]
            line[i_min:i_max] = 1
        data.append(line.reshape(-1, 1))

    img = np.hstack(data)

    print(img)
    print(index)

    with (output_path / "seiryo.npz").open("wb") as f:
        np.savez_compressed(f, img=img, index=index)


if __name__ == "__main__":
    main()
