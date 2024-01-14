from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from scipy import optimize
from sklearn import metrics


def load_silso_data(path: Path) -> pl.DataFrame:
    with path.open() as f:
        data = [i.split() for i in f.read().split("\n") if i]
    return (
        pl.LazyFrame(
            {
                "year": [int(i[0]) for i in data],
                "month": [int(i[1]) for i in data],
                "total": [float(i[3]) for i in data],
            },
        )
        .with_columns(
            pl.date(pl.col("year"), pl.col("month"), 1).alias("date"),
        )
        .drop("year", "month")
        .collect()
    )


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


def join_data(df_seiryo: pl.DataFrame, df_silso: pl.DataFrame) -> pl.DataFrame:
    df = (
        df_seiryo.lazy()
        .select("date", "total")
        .rename({"total": "seiryo"})
        .join(
            df_silso.lazy().select("date", "total").rename({"total": "silso"}),
            on="date",
            how="left",
        )
        .collect()
    )
    if df.filter(pl.col("silso").is_null()).height != 0:
        msg = "missing data for 'silso' on certain dates"
        raise ValueError(msg)
    return df


def calc_factor(df: pl.DataFrame) -> float:
    popt, _ = optimize.curve_fit(
        lambda x, a: x * a,
        df["silso"],
        df["seiryo"],
    )
    return popt[0]


def calc_r2(df: pl.DataFrame, factor: float) -> float:
    r2 = metrics.r2_score(
        df["seiryo"],
        df["silso"] * factor,
    )
    return float(r2)


def calc_ratio_and_diff(df: pl.DataFrame, factor: float) -> pl.DataFrame:
    return (
        df.lazy()
        .select(
            "date",
            (pl.col("seiryo") / pl.col("silso")).alias("ratio"),
            (pl.col("seiryo") / factor - pl.col("silso")).alias("diff"),
        )
        .sort("date")
        .collect()
    )


def draw_sunspot_number_whole_disk(df: pl.DataFrame) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(df["date"], df["seiryo"], lw=1, label="seiryo")
    ax.plot(df["date"], df["silso"], lw=1, label="SILSO")

    ax.set_title("seiryo's whole-disk sunspot number", y=1.1)
    ax.set_xlabel("date")
    ax.set_ylabel("sunspot number")

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


def draw_sunspot_number_hemispheric(df: pl.DataFrame) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(df["date"], df["north"], lw=1, label="north")
    ax.plot(df["date"], df["south"], lw=1, label="south")

    ax.set_title("seiryo's hemispheric sunspot number", y=1.1)
    ax.set_xlabel("date")
    ax.set_ylabel("sunspot number")

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


def draw_scatter(df: pl.DataFrame, factor: float, r2: float) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(
        [0, 180],
        np.poly1d([factor, 0])([0, 180]),
        lw=1,
        c="black",
        zorder=1,
    )
    ax.scatter(
        df["silso"],
        df["seiryo"],
        s=5,
        edgecolors="none",
        zorder=2,
    )
    ax.text(150, 40, f"$y={factor:.5f}x$")
    ax.text(150, 30, f"$R^2={r2:.5f}$")

    ax.set_title("SILSO and seiryo")
    ax.set_xlabel("SILSO")
    ax.set_ylabel("seiryo")

    fig.tight_layout()

    return fig


def draw_ratio_and_diff(df: pl.DataFrame, factor: float) -> Figure:
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.axhline(y=factor, c="black", lw=1, ls="--", zorder=1)
    ax1.plot(df["date"], df["ratio"], lw=1, zorder=2)

    ax1.set_title("ratio: seiryo / SILSO")
    ax1.set_xlabel("date")
    ax1.set_ylabel("ratio")
    ax1.grid()

    ax2.plot(df["date"], df["diff"], lw=1)

    ax2.set_title("difference: seiryo* - SILSO")
    ax2.set_xlabel("date")
    ax2.set_ylabel("difference")
    ax2.grid()

    fig.tight_layout()

    return fig


def main() -> None:
    path_seiryo = Path("out/seiryo/sn.parquet")
    path_silso = Path("data/SN_m_tot_V2.0.txt")
    output_path = Path("out/seiryo")

    df_seiryo = pl.read_parquet(path_seiryo).pipe(calc_sunspot_number)
    print(df_seiryo)
    df_silso = load_silso_data(path_silso)
    print(df_silso)

    df_joined = join_data(df_seiryo, df_silso)
    print(df_joined)

    fig1 = draw_sunspot_number_whole_disk(df_joined)
    for ext in "pdf", "png":
        file_path = output_path / f"sunspot_number_whole_disk.{ext}"
        fig1.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    fig2 = draw_sunspot_number_hemispheric(df_seiryo)
    for ext in "pdf", "png":
        file_path = output_path / f"sunspot_number_hemispheric.{ext}"
        fig2.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    factor = calc_factor(df_joined)
    print(f"{factor=}")

    r2 = calc_r2(df_joined, factor)
    print(f"{r2=}")

    fig3 = draw_scatter(df_joined, factor, r2)
    for ext in "pdf", "png":
        file_path = output_path / f"scatter.{ext}"
        fig3.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    df_ratio_diff = calc_ratio_and_diff(df_joined, factor)
    print(df_ratio_diff)

    fig4 = draw_ratio_and_diff(df_ratio_diff, factor)
    for ext in "pdf", "png":
        file_path = output_path / f"ratio_and_diff.{ext}"
        fig4.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    plt.show()


if __name__ == "__main__":
    main()
