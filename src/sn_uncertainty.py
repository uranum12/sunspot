from datetime import date
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dateutil.relativedelta import relativedelta
from uncertainty import errors as err


def to_year_fraction(dt: date) -> float:
    start_of_this_year = date(year=dt.year, month=1, day=1)
    start_of_next_year = start_of_this_year + relativedelta(years=1)

    year_elapsed = (dt - start_of_this_year).days
    year_duration = (start_of_next_year - start_of_this_year).days

    fraction = year_elapsed / year_duration

    return dt.year + fraction


def calc_sunspot_number(df: pl.LazyFrame) -> pl.DataFrame:
    return (
        df.drop("time", "remarks")
        .select(
            "date",
            (pl.col("ng") + pl.col("sg")).alias("ng"),  # groups
            (pl.col("nf") + pl.col("sf")).alias("ns"),  # spots
        )
        .with_columns(
            # wolf number
            # R = 10g + f
            (pl.col("ng").cast(pl.UInt16) * 10 + pl.col("ns")).alias("nc")
        )
        .sort("date")
        .collect()
    )


def main() -> None:
    path_uncertainty = Path("data/uncertainty/data_21_1947.zip")
    path_fujimori = Path("out/sn/all.parquet")
    output_path = Path("out/sn")

    # open other observer's data
    with ZipFile(path_uncertainty) as zf:
        with zf.open("station_names.txt") as f:
            station_names = [
                line.decode().removesuffix("\r\n") for line in f.readlines()
            ]
            station_names.append("Fujimori")
        with zf.open("fr_year.txt") as f:
            time = np.loadtxt(f.readlines())
        with zf.open("Ns.txt") as f:
            ns = np.loadtxt(f.readlines(), delimiter=",")

    # open fujimori's data
    df = calc_sunspot_number(pl.scan_parquet(path_fujimori))
    print(df)

    # calc the starting year of fujimori's data in fr_year
    start_fy = to_year_fraction(df.item(0, "date"))
    start = np.where(time == start_fy)[0][0]

    # if fujimori's data is longer than other observers
    # extend other observer's data with null
    if start + df.height - ns.shape[0] > 0:
        ns = np.vstack(
            [
                ns,
                np.full(
                    (start + df.height - ns.shape[0], ns.shape[1]), np.nan
                ),
            ]
        )

    # merge fujimori data with other observatories' data
    ns_repl = np.full(ns.shape[0], np.nan)
    ns_repl[start : start + len(df["date"])] = np.array(df["ns"])
    ns = np.append(ns, ns_repl.reshape(-1, 1), axis=1)
    print(ns)

    # short-term error (epsilon tilde)
    e1 = err.short_term_error(ns, period_rescaling=8)
    q75, q25 = np.nanpercentile(e1, [75, 25], axis=0)
    iqr_eps = q75 - q25
    print(iqr_eps)

    # long-term error (mu2)
    mu2 = err.long_term_error(ns, period_rescaling=8)
    q75, q25 = np.nanpercentile(mu2, [75, 25], axis=0)
    iqr_mu2 = q75 - q25
    print(iqr_mu2)

    colors = ["black"] * 21 + ["red"]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.scatter(iqr_eps, iqr_mu2, c=colors)

    ax.set_title("short-term and long-term error")
    ax.set_xlabel(r"IQR($\hat \widetilde{\epsilon}$)")
    ax.set_ylabel(r"IQR($\hat \mu_2$)")

    # codenames
    for i, txt in enumerate(station_names):
        ax.text(iqr_eps[i], iqr_mu2[i], txt, c=colors[i])

    fig.tight_layout()

    # save fig
    for ext in "pdf", "png":
        fig.savefig(
            output_path / f"uncertainty.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    plt.show()


if __name__ == "__main__":
    main()
