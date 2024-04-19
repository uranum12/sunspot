import json
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
            }
        )
        .with_columns(
            pl.date(pl.col("year"), pl.col("month"), 1).alias("date")
        )
        .drop("year", "month")
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


def calc_factor(df: pl.DataFrame) -> float:
    popt, _ = optimize.curve_fit(lambda x, a: x * a, df["silso"], df["seiryo"])
    return popt[0]


def calc_r2(df: pl.DataFrame, factor: float) -> float:
    r2 = metrics.r2_score(df["seiryo"], df["silso"] * factor)
    return float(r2)


def draw_sunspot_number_with_silso(df: pl.DataFrame) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(df["date"], df["seiryo"], lw=1, label="seiryo")
    ax.plot(df["date"], df["silso"], lw=1, label="SILSO")

    ax.set_title("seiryo's whole-disk sunspot number", y=1.1)
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


def draw_scatter(df: pl.DataFrame, factor: float, r2: float) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(
        [0, 180], np.poly1d([factor, 0])([0, 180]), lw=1, c="black", zorder=1
    )
    ax.scatter(df["silso"], df["seiryo"], s=5, edgecolors="none", zorder=2)
    ax.text(150, 40, f"$y={factor:.5f}x$")
    ax.text(150, 30, f"$R^2={r2:.5f}$")

    ax.set_title("SILSO and seiryo")
    ax.set_xlabel("SILSO")
    ax.set_ylabel("seiryo")

    ax.grid()

    fig.tight_layout()

    return fig


def draw_ratio(df: pl.DataFrame, factor: float) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.axhline(y=factor, c="black", lw=1, ls="--", zorder=1)
    ax.plot(df["date"], df["ratio"], lw=1, zorder=2)

    ax.set_title("ratio: seiryo / SILSO")
    ax.set_xlabel("date")
    ax.set_ylabel("ratio")
    ax.grid()

    fig.tight_layout()

    return fig


def draw_diff(df: pl.DataFrame) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(df["date"], df["diff"], lw=1)

    ax.set_title("difference: seiryo* - SILSO")
    ax.set_xlabel("date")
    ax.set_ylabel("difference")
    ax.grid()

    fig.tight_layout()

    return fig


def draw_ratio_diff_1(df: pl.DataFrame, factor: float) -> Figure:
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    ax1.axhline(y=factor, c="black", lw=1, ls="--", zorder=1)
    ax1.plot(df["date"], df["ratio"], lw=1, zorder=2)

    ax1.set_title("ratio: seiryo / SILSO")
    ax1.set_ylabel("ratio")
    ax1.tick_params(bottom=False, labelbottom=False)
    ax1.grid()

    ax2.plot(df["date"], df["diff"], lw=1)

    ax2.set_title("difference: seiryo* - SILSO")
    ax2.set_xlabel("date")
    ax2.set_ylabel("difference")
    ax2.grid()

    fig.tight_layout()

    return fig


def draw_ratio_diff_2(df: pl.DataFrame) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(111)

    ax1.plot(df["date"], df["ratio"], c="C0", lw=1, label="ratio")

    ax1.set_title(
        "ratio: seiryo / SILSO and difference: seiryo* - SILSO", y=1.1
    )
    ax1.set_xlabel("date")
    ax1.set_ylabel("ratio")
    ax1.grid()

    ax2 = ax1.twinx()

    ax2.plot(df["date"], df["diff"], c="C1", lw=1, label="diff")  # type: ignore[attr-defined]
    ax2.set_ylabel("difference")
    ax2.grid()

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()  # type: ignore[attr-defined]
    ax1.legend(
        h1 + h2,
        l1 + l2,
        fancybox=False,
        edgecolor="black",
        framealpha=1,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0,
        ncol=2,
    )

    return fig


def main() -> None:
    path_seiryo = Path("out/seiryo/sunspot/monthly.parquet")
    path_silso = Path("data/SN_m_tot_V2.0.txt")
    output_path = Path("out/seiryo/sunspot")

    df_seiryo = pl.read_parquet(path_seiryo)
    print(df_seiryo)

    df_silso = load_silso_data(path_silso)
    print(df_silso)

    df_seiryo_with_silso = join_data(df_seiryo, df_silso)
    print(df_seiryo_with_silso)
    df_seiryo_with_silso.write_parquet(output_path / "with_silso.parquet")

    factor = calc_factor(df_seiryo_with_silso)
    print(f"{factor=}")

    r2 = calc_r2(df_seiryo_with_silso, factor)
    print(f"{r2=}")

    with (output_path / "factor_r2.json").open("w") as json_file:
        json.dump({"factor": factor, "r2": r2}, json_file)

    df_ratio_and_diff = calc_ratio_and_diff(df_seiryo_with_silso, factor)
    print(df_ratio_and_diff)
    df_ratio_and_diff.write_parquet(output_path / "ratio_diff.parquet")

    fig_with_silso = draw_sunspot_number_with_silso(df_seiryo_with_silso)

    for f in ["png", "pdf"]:
        fig_with_silso.savefig(
            output_path / f"with_silso.{f}",
            format=f,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    fig_scatter = draw_scatter(df_seiryo_with_silso, factor, r2)

    for f in ["png", "pdf"]:
        fig_scatter.savefig(
            output_path / f"scatter.{f}",
            format=f,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    fig_ratio = draw_ratio(df_ratio_and_diff, factor)

    for f in ["png", "pdf"]:
        fig_ratio.savefig(
            output_path / f"ratio.{f}",
            format=f,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    fig_diff = draw_diff(df_ratio_and_diff)

    for f in ["png", "pdf"]:
        fig_diff.savefig(
            output_path / f"diff.{f}",
            format=f,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
    fig_ratio_and_diff_1 = draw_ratio_diff_1(df_ratio_and_diff, factor)

    for f in ["png", "pdf"]:
        fig_ratio_and_diff_1.savefig(
            output_path / f"ratio_diff_1.{f}",
            format=f,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    fig_ratio_and_diff_2 = draw_ratio_diff_2(df_ratio_and_diff)

    for f in ["png", "pdf"]:
        fig_ratio_and_diff_2.savefig(
            output_path / f"ratio_diff_2.{f}",
            format=f,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )


if __name__ == "__main__":
    main()
