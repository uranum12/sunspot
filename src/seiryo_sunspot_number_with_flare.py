from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure


def load_flare_data(path: Path) -> pl.DataFrame:
    dates: list[date] = []
    indexes: list[float] = []
    for file in path.glob("*.txt"):
        year = int(file.name[18:22])
        with file.open("r") as f:
            l_mean = next(
                line.strip()
                for line in f.readlines()
                if line.startswith("Mean")
            )
        for i, d in enumerate(l_mean.split()[1:]):
            dates.append(date(year, i + 1, 1))
            indexes.append(float(d))

    return pl.DataFrame({"date": dates, "index": indexes}).sort("date")


def join_data(df_sn: pl.DataFrame, df_flare: pl.DataFrame) -> pl.DataFrame:
    return (
        df_sn.lazy()
        .select("date", "total")
        .rename({"total": "seiryo"})
        .join(
            df_flare.lazy().rename({"index": "flare"}), on="date", how="left"
        )
        .collect()
    )


def draw_sunspot_number_with_flare(df: pl.DataFrame) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(111)

    ax1.plot(df["date"], df["seiryo"], c="C0", lw=1, label="seiryo")

    ax1.set_title("sunspot number and solar flare index", y=1.1)
    ax1.set_xlabel("date")
    ax1.set_ylabel("sunspot number")
    ax1.grid()

    ax2 = ax1.twinx()

    ax2.plot(df["date"], df["flare"], c="C1", lw=1, label="flare")  # type: ignore[attr-defined]
    ax2.set_ylabel("solar flare index")
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

    fig.tight_layout()

    return fig


def main() -> None:
    path_seiryo = Path("out/seiryo/sunspot/monthly.parquet")
    path_flare = Path("data/flare")
    output_path = Path("out/seiryo/sunspot")

    df_seiryo = pl.read_parquet(path_seiryo)
    print(df_seiryo)

    df_flare = load_flare_data(path_flare)
    print(df_flare)

    df_with_flare = join_data(df_seiryo, df_flare)
    print(df_with_flare)
    df_with_flare.write_parquet(output_path / "with_flare.parquet")

    fig_with_flare = draw_sunspot_number_with_flare(df_with_flare)

    for f in ["png", "pdf"]:
        fig_with_flare.savefig(
            output_path / f"with_flare.{f}",
            format=f,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )


if __name__ == "__main__":
    main()
