from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def main() -> None:
    data_file = Path("out/wolf/fujimori.parquet")
    output_path = Path("out/wolf")
    output_path.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(data_file)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.scatter(df["date"], df["nr"], s=1, edgecolors="none", label="north")
    ax.scatter(df["date"], df["sr"], s=1, edgecolors="none", label="south")
    ax.scatter(df["date"], df["tr"], s=1, edgecolors="none", label="total")

    ax.set_title("relative sunspot number")
    ax.set_xlabel("date")
    ax.set_ylabel("sunspot number")

    ax.legend(fancybox=False, edgecolor="black", framealpha=1, markerscale=5)

    for ext in "pdf", "png":
        fig.savefig(
            output_path / f"fujimori_daily.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    plt.show()


if __name__ == "__main__":
    main()
