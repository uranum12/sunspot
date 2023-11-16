from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import optimize
from sklearn import metrics


def main() -> None:  # noqa: PLR0915
    path_fujimori = Path("out/sn/all.parquet")
    path_silso = Path("data/SN_m_tot_V2.0.txt")
    output_path = Path("out/sn")

    df_fujimori = (
        pl.scan_parquet(path_fujimori)
        .drop("time", "remarks")
        .drop_nulls()
        .with_columns(
            (pl.col("ng") + pl.col("sg")).alias("tg"),
            (pl.col("nf") + pl.col("sf")).alias("tf"),
        )
        .with_columns(
            # R = 10g + f
            (pl.col("ng").cast(pl.UInt16) * 10 + pl.col("nf")).alias("nr"),
            (pl.col("sg").cast(pl.UInt16) * 10 + pl.col("sf")).alias("sr"),
            (pl.col("tg").cast(pl.UInt16) * 10 + pl.col("tf")).alias("tr"),
        )
        .with_columns(pl.col("date").dt.truncate("1mo"))
        .group_by("date")
        .mean()
        .sort("date")
        .collect()
    )
    print(df_fujimori)

    with path_silso.open() as f:
        data = [i.split() for i in f.read().split("\n") if i]
    df_silso = (
        pl.DataFrame(
            {
                "year": [int(i[0]) for i in data],
                "month": [int(i[1]) for i in data],
                "silso": [float(i[3]) for i in data],
            },
        )
        .with_columns(
            pl.date(pl.col("year"), pl.col("month"), 1).alias("date"),
        )
        .drop("year", "month")
    )
    print(df_silso)

    df_joined = (
        df_fujimori.select("date", "tr")
        .rename({"tr": "fujimori"})
        .join(df_silso, on="date")
    )
    print(df_joined)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(df_joined["date"], df_joined["fujimori"], lw=1, label="fujimori")
    ax.plot(df_joined["date"], df_joined["silso"], lw=1, label="SILSO")

    ax.set_title("fujimori's sunspot number", y=1.1)
    ax.set_xlabel("year")
    ax.set_ylabel("relative sunspot number")

    ax.legend(
        fancybox=False,
        edgecolor="black",
        framealpha=1,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            1.02,
        ),
        borderaxespad=0,
        ncol=2,
    )

    fig.tight_layout()

    for ext in "pdf", "png":
        fig.savefig(
            output_path / f"sunspot_number.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    fig2 = plt.figure(figsize=(8, 5))
    ax = fig2.add_subplot(111)

    ax.plot(df_fujimori["date"], df_fujimori["nr"], lw=1, label="north")
    ax.plot(df_fujimori["date"], df_fujimori["sr"], lw=1, label="south")

    ax.set_title("fujimori's sunspot number", y=1.1)
    ax.set_xlabel("year")
    ax.set_ylabel("relative sunspot number")

    ax.legend(
        fancybox=False,
        edgecolor="black",
        framealpha=1,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            1.02,
        ),
        borderaxespad=0,
        ncol=2,
    )

    for ext in "pdf", "png":
        fig2.savefig(
            output_path / f"sunspot_number_sn.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    popt, _ = optimize.curve_fit(
        lambda x, a: x * a,
        df_joined["silso"],
        df_joined["fujimori"],
    )
    print(f"{popt[0]=}")
    r2 = float(
        metrics.r2_score(
            df_joined["fujimori"],
            df_joined["silso"] * popt[0],
        ),
    )
    print(f"{r2=}")

    fig3 = plt.figure(figsize=(8, 5))
    ax = fig3.add_subplot(111)

    ax.plot(
        [0, 360],
        np.poly1d([r2, 0])([0, 360]),
        lw=1,
        c="black",
        zorder=1,
    )
    ax.scatter(
        df_joined["silso"],
        df_joined["fujimori"],
        s=3,
        edgecolors="none",
        zorder=2,
    )
    ax.text(300, 160, f"$y={popt[0]:.5f}x$")
    ax.text(300, 140, f"$R^2={r2:.5f}$")

    ax.set_title("SILSO and fujimori")
    ax.set_xlabel("SILSO")
    ax.set_ylabel("fujimori")

    for ext in "pdf", "png":
        fig3.savefig(
            output_path / f"scatter.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    df_ratio_diff = (
        df_joined.lazy()
        .with_columns(
            (pl.col("fujimori") / pl.col("silso")).alias("ratio"),
            (pl.col("fujimori") * r2 - pl.col("silso")).alias("diff"),
        )
        .sort("date")
        .collect()
    )
    print(df_ratio_diff)

    fig4 = plt.figure(figsize=(8, 8))
    ax1 = fig4.add_subplot(211)
    ax2 = fig4.add_subplot(212)

    ax1.axhline(y=r2, c="black", lw=1, ls="--", zorder=1)
    ax1.plot(df_ratio_diff["date"], df_ratio_diff["ratio"], lw=1, zorder=2)

    ax1.set_title("ratio: fujimori / SILSO")
    ax1.set_xlabel("year")
    ax1.set_ylabel("ratio")
    ax1.grid()

    ax2.plot(df_ratio_diff["date"], df_ratio_diff["diff"], lw=1)

    ax2.set_title("difference: fujimori * $R^2$ - SILSO")
    ax2.set_xlabel("year")
    ax2.set_ylabel("difference")
    ax2.grid()

    fig4.tight_layout()

    for ext in "pdf", "png":
        fig4.savefig(
            output_path / f"ratio_diff.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    plt.show()


if __name__ == "__main__":
    main()
