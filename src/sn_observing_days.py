from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def main() -> None:
    data_file = Path("out/sn/all.parquet")
    output_path = Path("out/sn")

    df = (
        pl.scan_parquet(data_file)
        .filter(
            pl.all_horizontal(pl.col("ng", "nf", "sg", "sf").is_not_null()),
        )
        .select("date")
        .with_columns(pl.col("date").dt.truncate("1y").dt.year())
        .group_by("date")
        .len()
        .sort("date")
        .select(
            pl.col("date"),
            pl.when(pl.col("date").lt(1971))
            .then(pl.col("len"))
            .alias("100mm"),
            pl.when(pl.col("date").ge(1971)).then(pl.col("len")).alias("80mm"),
        )
        .collect()
    )
    print(df)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.bar(df["date"], df["100mm"], label="100mm reflection x56 direct vision")
    ax.bar(df["date"], df["80mm"], label="80mm refraction x67 projection")

    ax.set_title("observing days per year", y=1.1)
    ax.set_xlabel("year")
    ax.set_ylabel("observing days")

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

    for ext in "pdf", "png":
        fig.savefig(
            output_path / f"observing_days.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    plt.show()


if __name__ == "__main__":
    main()
