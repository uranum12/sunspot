from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dateutil.relativedelta import relativedelta

import butterfly_common

if TYPE_CHECKING:
    from datetime import date


def main() -> None:
    data_file = Path("out/ar/all.parquet")
    output_path = Path("out/butterfly")
    output_path.mkdir(parents=True, exist_ok=True)

    df_file = (
        pl.scan_parquet(data_file)
        .with_columns(pl.col("lat_left", "lat_right").cast(pl.Int8))
        .pipe(butterfly_common.reverse_south)
        .pipe(butterfly_common.reverse_minus)
        .pipe(butterfly_common.fix_order)
        .pipe(butterfly_common.extract_date)
    )

    start, end = butterfly_common.calc_start_end(df_file, replace=True)

    data: list[np.ndarray] = []
    index: list[date] = []

    current = start
    while current <= end:
        df = (
            butterfly_common.filter_data(df_file, current)
            .rename({"lat_left": "min", "lat_right": "max"})
            .collect()
        )

        line = np.zeros(201, dtype=np.uint8)
        for i in df.iter_rows(named=True):
            i_min = 100 + 2 * i["min"]
            i_max = 100 + 2 * i["max"]
            line[i_min:i_max] = 1
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
            output_path / f"fujimori_monthly.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    plt.show()


if __name__ == "__main__":
    main()