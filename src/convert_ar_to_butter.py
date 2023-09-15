from datetime import date
from pathlib import Path

import polars as pl
from dateutil.relativedelta import relativedelta

df = pl.scan_parquet(Path("out/ar/all.parquet")).select(
    "ns",
    pl.when(pl.col("lat_left") > pl.col("lat_right"))
    .then(pl.col("lat_right"))
    .otherwise(pl.col("lat_left"))
    .alias("lat_left"),
    pl.when(pl.col("lat_left") > pl.col("lat_right"))
    .then(pl.col("lat_left"))
    .otherwise(pl.col("lat_right"))
    .alias("lat_right"),
    pl.col("first").dt.year().suffix("_year"),
    pl.col("first").dt.month().suffix("_month"),
    pl.col("last").dt.year().suffix("_year"),
    pl.col("last").dt.month().suffix("_month"),
)

start = date(1953, 3, 1)
end = date(2016, 6, 1)

with Path("out/butter.txt").open("w") as file:
    file.write("//Data File for Butterfly Diagram\n")
    file.write(f">>{start.year}/{start.month:0{2}}-")
    file.write(f"{end.year}/{end.month:0{2}}\n\n")
    file.write("<----data---->\n")

    current = start
    while current <= end:
        year = current.year
        month = current.month
        current += relativedelta(months=1)

        for ns in "N", "S":
            data = (
                df.filter(
                    (
                        pl.col("first_year").eq(year)
                        & pl.col("first_month").eq(month)
                    )
                    | (
                        pl.col("last_year").eq(year)
                        & pl.col("last_month").eq(month)
                    ),
                )
                .filter(pl.col("ns").eq(ns))
                .select("lat_left", "lat_right")
                .unique()
                .drop_nulls()
                .sort("lat_left", "lat_right")
                .collect()
            )
            merged: list[list[int]] = []
            for row in data.iter_rows(named=True):
                if len(merged) == 0:
                    merged.append([row["lat_left"], row["lat_right"]])
                    continue
                if row["lat_left"] < merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], row["lat_right"])
                else:
                    merged.append([row["lat_left"], row["lat_right"]])
            item = " ".join([f"{i[0]}-{i[1]}" for i in merged])
            file.write(f"{year}/{month:0{2}}/{ns}:{item}\n")
