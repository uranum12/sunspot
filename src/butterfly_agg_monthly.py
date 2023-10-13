from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

import butterfly_agg_common
import butterfly_common


def main() -> None:
    data_file = Path("out/ar/all.parquet")
    output_path = Path("out/butterfly")
    output_path.mkdir(parents=True, exist_ok=True)

    df_file = (
        pl.scan_parquet(data_file)
        .pipe(butterfly_common.cast_lat_sign)
        .pipe(butterfly_common.drop_lat_null)
        .pipe(butterfly_common.reverse_south)
        .pipe(butterfly_common.reverse_minus)
        .pipe(butterfly_common.fix_order)
        .pipe(butterfly_common.complement_last)
        .pipe(butterfly_common.truncate_day)
        .collect()
    )

    start, end = butterfly_common.calc_start_end(df_file.lazy(), replace=True)

    lat_max = 50
    lat_min = -50

    data: list[npt.NDArray[np.uint8]] = []
    date_index = butterfly_agg_common.create_date_index_monthly(start, end)
    lat_index = butterfly_agg_common.create_lat_index(lat_min, lat_max)

    for i in date_index:
        df = (
            df_file.lazy()
            .pipe(butterfly_common.filter_data, date=i.astype(date))
            .select("lat_min", "lat_max")
            .collect()
        )

        df_data = df.to_dict(as_series=False)
        line = butterfly_agg_common.create_line(
            df_data["lat_min"],
            df_data["lat_max"],
            lat_min,
            lat_max,
        )
        data.append(line.reshape(-1, 1))

    img = np.hstack(data)

    print(img)
    print(date_index)
    print(lat_index)

    with (output_path / "fujimori_monthly.npz").open("wb") as f:
        np.savez_compressed(f, img=img, date=date_index, lat=lat_index)


if __name__ == "__main__":
    main()
