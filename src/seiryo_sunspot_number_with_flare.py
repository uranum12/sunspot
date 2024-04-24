from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure


def load_flare_file(path: Path) -> pl.DataFrame:
    with path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]
    year = int(lines[3])
    mean_line = [
        line.replace("Mean", "").split()
        for line in lines
        if line.startswith("Mean")
    ]
    dates: list[date] = []
    indexes: list[float] = []
    for i, d in enumerate(mean_line[0]):
        dates.append(date(year, i + 1, 1))
        indexes.append(float(d))
    return pl.DataFrame({"date": dates, "index": indexes})


def load_flare_data(path: Path) -> pl.DataFrame:
    df_north = pl.concat(
        load_flare_file(file).lazy() for file in path.glob("*north*.txt")
    ).rename({"index": "north"})
    df_south = pl.concat(
        load_flare_file(file).lazy() for file in path.glob("*south*.txt")
    ).rename({"index": "south"})
    df_total = pl.concat(
        load_flare_file(file).lazy() for file in path.glob("*total*.txt")
    ).rename({"index": "total"})
    return (
        df_north.join(df_south, on="date", how="outer_coalesce")
        .join(df_total, on="date", how="outer_coalesce")
        .sort("date")
        .collect()
    )


def join_data(df_seiryo: pl.DataFrame, df_flare: pl.DataFrame) -> pl.DataFrame:
    return (
        df_seiryo.lazy()
        .rename(
            {
                "north": "seiryo_north",
                "south": "seiryo_south",
                "total": "seiryo_total",
            }
        )
        .join(
            df_flare.lazy().rename(
                {
                    "north": "flare_north",
                    "south": "flare_south",
                    "total": "flare_total",
                }
            ),
            on="date",
            how="left",
        )
        .collect()
    )


def draw_sunspot_number_with_flare(df: pl.DataFrame) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(111)

    ax1.plot(df["date"], df["seiryo_total"], c="C0", lw=1, label="seiryo")

    ax1.set_title("sunspot number and solar flare index", y=1.1)
    ax1.set_xlabel("date")
    ax1.set_ylabel("sunspot number")
    ax1.grid()

    ax2 = ax1.twinx()

    ax2.plot(df["date"], df["flare_total"], c="C1", lw=1, label="flare")  # type: ignore[attr-defined]
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


def draw_sunspot_number_with_flare_hemispheric(df: pl.DataFrame) -> Figure:
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    ax1_twin = ax1.twinx()

    ax1.plot(df["date"], df["seiryo_north"], lw=1, c="C0", label="seiryo")
    ax1_twin.plot(df["date"], df["flare_north"], lw=1, c="C1", label="flare")  # type: ignore[attr-defined]

    ax1.set_title("north", y=1.1)
    ax1.set_ylabel("sunspot number")
    ax1_twin.set_ylabel("solar flare index")
    ax1.tick_params(bottom=False, labelbottom=False)
    ax1.grid()
    ax1_twin.grid()

    h1, l1 = ax1.get_legend_handles_labels()
    h1_twin, l1_twin = ax1_twin.get_legend_handles_labels()  # type: ignore[attr-defined]
    ax1.legend(
        h1 + h1_twin,
        l1 + l1_twin,
        fancybox=False,
        edgecolor="black",
        framealpha=1,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0,
        ncol=2,
    )

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2_twin = ax2.twinx()

    ax2.plot(df["date"], df["seiryo_south"], lw=1, c="C0", label="seiryo")
    ax2_twin.plot(df["date"], df["flare_south"], lw=1, c="C1", label="flare")  # type: ignore[attr-defined]

    ax2.set_title("south", y=1.1)
    ax2.set_ylabel("sunspot number")
    ax2_twin.set_ylabel("solar flare index")
    ax2.set_xlabel("date")
    ax2.grid()
    ax2_twin.grid()

    h2, l2 = ax2.get_legend_handles_labels()
    h2_twin, l2_twin = ax2_twin.get_legend_handles_labels()  # type: ignore[attr-defined]
    ax2.legend(
        h2 + h2_twin,
        l2 + l2_twin,
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

    df_flare = load_flare_data(path_flare).sort("date")
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

    fig_with_flare_hemispheric = draw_sunspot_number_with_flare_hemispheric(
        df_with_flare
    )

    for f in ["png", "pdf"]:
        fig_with_flare_hemispheric.savefig(
            output_path / f"with_flare_hemispheric.{f}",
            format=f,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )


if __name__ == "__main__":
    main()
