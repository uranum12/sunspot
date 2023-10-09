from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from dateutil.relativedelta import relativedelta

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
        .pipe(butterfly_common.extract_date)
        .collect()
    )

    start, end = butterfly_common.calc_start_end(df_file.lazy(), replace=True)

    data: list[npt.NDArray[np.uint8]] = []
    index = np.arange(
        start,
        end + relativedelta(months=1),
        dtype="datetime64[M]",
    )

    for current in (i.astype(date) for i in index):
        df = (
            df_file.lazy()
            .pipe(butterfly_common.filter_data_monthly, date=current)
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

    with (output_path / "fujimori_monthly.npz").open("wb") as f:
        np.savez_compressed(f, img=img, index=index)


if __name__ == "__main__":
    main()
