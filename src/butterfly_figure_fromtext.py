from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dateutil.relativedelta import relativedelta

if TYPE_CHECKING:
    from datetime import date


def main() -> None:
    data_file = Path("data/seiryo/1950-2023.txt")
    output_path = Path("out/butterfly")
    output_path.mkdir(parents=True, exist_ok=True)

    with data_file.open() as f:
        df_file = (
            pl.DataFrame({"txt": f.readlines()[4:]})
            .lazy()
            .select(
                pl.col("txt").str.extract(r"^(\d+/\d+)/").alias("date"),
                pl.col("txt").str.extract(r"/([NS])").alias("ns"),
                pl.col("txt").str.extract_all(r"(\d+-\d+)").alias("data"),
            )
            .with_columns(
                pl.col("date").str.strptime(pl.Date, "%Y/%m"),
                pl.when(pl.col("ns") == "N")
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
            .group_by("date")
            .agg(pl.col("data").flatten().drop_nulls())
            .explode("data")
            .with_columns(
                pl.col("data").list.get(0).alias("min"),
                pl.col("data").list.get(1).alias("max"),
            )
            .drop("data")
            .collect()
        )

    start = df_file.select("date").min().item()
    end = df_file.select("date").max().item()
    print(start, end)

    data: list[np.ndarray] = []
    index: list[date] = []

    current: date = start
    while current <= end:
        df = (
            df_file.lazy()
            .filter(pl.col("date").eq(current))
            .select("min", "max")
            .unique()
            .drop_nulls()
            .collect()
        )

        line = np.zeros(201, dtype=np.uint8)
        for i in df.iter_rows(named=True):
            i_min = 100 + 2 * i["min"]
            i_max = 100 + 2 * i["max"]
            line[i_min:i_max] = 1
        print(line)
        data.append(line.reshape(-1, 1))

        index.append(current)

        current += relativedelta(months=1)

    img = np.hstack(data)

    label = {
        i: str(d.year)
        for i, d in enumerate(index)
        if d.month == 1 and d.year % 10 == 0
    }

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap="binary")

    ax.set_title("butterfly diagram")

    ax.set_xlabel("date")
    ax.set_xticks(list(label.keys()))
    ax.set_xticklabels(label.values())

    ax.set_ylabel("latitude")
    ax.set_yticks(range(0, 200 + 1, 20))
    ax.set_yticklabels(str(abs(x)) for x in range(-50, 50 + 1, 10))

    for ext in "pdf", "png":
        fig.savefig(
            output_path / f"seiryo.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    plt.show()


if __name__ == "__main__":
    main()
