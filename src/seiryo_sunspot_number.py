from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure


def calc_sunspot_number(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.lazy()
        .select(
            pl.col("date").dt.truncate("1mo"),
            (pl.col("ng").cast(pl.UInt16) * 10 + pl.col("nf")).alias("north"),
            (pl.col("sg").cast(pl.UInt16) * 10 + pl.col("sf")).alias("south"),
            (pl.col("tg").cast(pl.UInt16) * 10 + pl.col("tf")).alias("total"),
        )
        .group_by("date")
        .mean()
        .sort("date")
        .collect()
    )


def draw_sunspot_number_whole_disk(df: pl.DataFrame) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(df["date"], df["total"])

    ax.set_title("seiryo's whole-disk sunspot number")
    ax.set_xlabel("date")
    ax.set_ylabel("sunspot number")

    ax.grid()

    fig.tight_layout()

    return fig


def draw_sunspot_number_hemispheric(df: pl.DataFrame) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(df["date"], df["north"], lw=1, label="north")
    ax.plot(df["date"], df["south"], lw=1, label="south")

    ax.set_title("seiryo's hemispheric sunspot number", y=1.1)
    ax.set_xlabel("date")
    ax.set_ylabel("sunspot number")

    ax.grid()
    ax.legend(
        fancybox=False,
        edgecolor="black",
        framealpha=1,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0,
        ncol=2,
    )

    fig.tight_layout()

    return fig


def main() -> None:
    path_seiryo = Path("out/seiryo/sn.parquet")
    output_path = Path("out/seiryo/sunspot")
    output_path.mkdir(exist_ok=True)

    df = pl.read_parquet(path_seiryo)
    df = calc_sunspot_number(df)
    print(df)
    df.write_parquet(output_path / "monthly.parquet")

    fig_whole_disk = draw_sunspot_number_whole_disk(df)

    for f in ["png", "pdf"]:
        fig_whole_disk.savefig(
            output_path / f"whole_disk.{f}",
            format=f,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    fig_hemispheric = draw_sunspot_number_hemispheric(df)

    for f in ["png", "pdf"]:
        fig_hemispheric.savefig(
            output_path / f"hemispheric.{f}",
            format=f,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )


if __name__ == "__main__":
    main()
