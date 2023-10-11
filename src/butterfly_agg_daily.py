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
        .pipe(butterfly_common.cast_lat_sign)
        .pipe(butterfly_common.reverse_south)
        .pipe(butterfly_common.reverse_minus)
        .pipe(butterfly_common.fix_order)
        .collect()
    )

    start, end = butterfly_common.calc_start_end(df_file.lazy())

    lat_n_max = 50
    lat_s_max = 50

    data: list[npt.NDArray[np.uint8]] = []
    date_index = np.arange(
        start,
        end + timedelta(days=1),
        dtype="datetime64[D]",
    )
    lat_index = np.insert(
        np.abs(np.arange(-lat_n_max, lat_s_max + 1, dtype=np.int8)),
        np.arange(1, lat_n_max + lat_s_max + 1),
        -1,
    )

    for current in (i.astype(date) for i in date_index):
        df = (
            df_file.lazy()
            .pipe(butterfly_common.filter_data_daily, date=current)
            .collect()
        )

        line = np.zeros(2 * (lat_n_max + lat_s_max) + 1, dtype=np.uint8)
        for i in df.iter_rows(named=True):
            i_min = 2 * (lat_n_max + i["lat_left"])
            i_max = 2 * (lat_n_max + i["lat_right"]) + 1
            line[i_min:i_max] = 1
        data.append(line.reshape(-1, 1))

    img = np.hstack(data)

    print(img)
    print(date_index)
    print(lat_index)

    with (output_path / "fujimori_daily.npz").open("wb") as f:
        np.savez_compressed(f, img=img, date=date_index, lat=lat_index)


if __name__ == "__main__":
    main()
