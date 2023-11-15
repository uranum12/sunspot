from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def main() -> None:
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

    df = df_fujimori.select("date", "tr").join(df_silso, on="date")
    print(df)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(df["date"], df["tr"], lw=1, label="fujimori")
    ax.plot(df["date"], df["silso"], lw=1, label="SILSO")

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

    fig3 = plt.figure(figsize=(8, 5))
    ax = fig3.add_subplot(111)

    ax.scatter(
        df["silso"],
        df["tr"],
        s=3,
        edgecolors="none",
    )

    ax.set_title("silso and fujimori")
    ax.set_xlabel("silso")
    ax.set_ylabel("fujimori")

    plt.show()


if __name__ == "__main__":
    main()
