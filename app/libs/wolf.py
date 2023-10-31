import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure


def calc_lat(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        ((pl.col("lat_min") + pl.col("lat_max")) / 2).alias("lat"),
    )


def daily(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.group_by("date")
        .agg(
            pl.col("num").filter(pl.col("lat") >= 0).alias("n"),
            pl.col("num").filter(pl.col("lat") < 0).alias("s"),
            pl.col("num").count().cast(pl.UInt8).alias("tg"),
            pl.col("num").sum().cast(pl.UInt16).alias("tf"),
        )
        .with_columns(
            pl.col("n").list.len().cast(pl.UInt8).alias("ng"),
            pl.col("n").list.sum().cast(pl.UInt16).alias("nf"),
            pl.col("s").list.len().cast(pl.UInt8).alias("sg"),
            pl.col("s").list.sum().cast(pl.UInt16).alias("sf"),
        )
        .drop("n", "s")
        .with_columns(
            (pl.col("ng").cast(pl.UInt16) * 10 + pl.col("nf")).alias("nr"),
            (pl.col("sg").cast(pl.UInt16) * 10 + pl.col("sf")).alias("sr"),
            (pl.col("tg").cast(pl.UInt16) * 10 + pl.col("tf")).alias("tr"),
        )
        .select(
            ["date", "ng", "nf", "nr", "sg", "sf", "sr", "tg", "tf", "tr"],
        )
        .sort("date")
    )


def monthly(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.with_columns(pl.col("date").dt.truncate("1mo"))
        .group_by("date")
        .mean()
    )


def graph_daily(df: pl.DataFrame) -> Figure:
    fig, ax = plt.subplots()

    ax.scatter(df["date"], df["nr"], s=1, edgecolors="none", label="north")
    ax.scatter(df["date"], df["sr"], s=1, edgecolors="none", label="south")
    ax.scatter(df["date"], df["tr"], s=1, edgecolors="none", label="total")

    ax.set_title("relative sunspot number")
    ax.set_xlabel("date")
    ax.set_ylabel("sunspot number")
    ax.legend(fancybox=False, edgecolor="black", framealpha=1)

    return fig


def graph_monthly(df: pl.DataFrame) -> Figure:
    fig, ax = plt.subplots()

    ax.plot(df["date"], df["nr"], lw=1, label="north")
    ax.plot(df["date"], df["sr"], lw=1, label="south")
    ax.plot(df["date"], df["tr"], lw=1, label="total")

    ax.set_title("relative sunspot number")
    ax.set_xlabel("date")
    ax.set_ylabel("sunspot number")
    ax.legend(fancybox=False, edgecolor="black", framealpha=1)

    return fig
