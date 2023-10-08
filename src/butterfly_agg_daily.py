from datetime import date, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

import butterfly_common


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
        .collect()
    )

    start, end = butterfly_common.calc_start_end(df_file.lazy())

    data: list[npt.NDArray[np.uint8]] = []
    index = np.arange(
        start,
        end + timedelta(days=1),
        dtype="datetime64[D]",
    )

    for current in (i.astype(date) for i in index):
        df = (
            df_file.lazy()
            .filter(
                pl.when(pl.col("last").is_null())
                .then(pl.col("first").eq(current.replace(day=1)))
                .otherwise(
                    pl.lit(current).is_between(
                        pl.col("first"),
                        pl.col("last"),
                    ),
                ),
            )
            .select("lat_left", "lat_right")
            .drop_nulls()
            .collect()
        )

        line = np.zeros(201, dtype=np.uint8)
        for i in df.iter_rows(named=True):
            i_min = 100 + 2 * i["lat_left"]
            i_max = 100 + 2 * i["lat_right"]
            line[i_min:i_max] = 1
        data.append(line.reshape(-1, 1))

    img = np.hstack(data)

    print(img)
    print(index)

    with (output_path / "fujimori_daily.npz").open("wb") as f:
        np.savez_compressed(f, img=img, index=index)


if __name__ == "__main__":
    main()
