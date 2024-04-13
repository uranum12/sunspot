from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go
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

    ax.set_title("seiryo's whole-disk sunspot number", y=1.1)
    ax.set_xlabel("date")
    ax.set_ylabel("sunspot number")

    fig.tight_layout()

    return fig


def draw_sunspot_number_whole_disk_plotly(df: pl.DataFrame) -> go.Figure:
    return (
        go.Figure()
        .add_trace(
            go.Scatter(x=df["date"], y=df["total"], mode="lines+markers")
        )
        .update_layout(
            {
                "title": {"text": "seiryo's whole-disk sunspot number"},
                "xaxis": {"title": {"text": "date"}},
                "yaxis": {"title": {"text": "sunspot number"}},
            }
        )
    )


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


def draw_sunspot_number_hemispheric_plotly(df: pl.DataFrame) -> go.Figure:
    return (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=df["date"], y=df["north"], mode="lines+markers", name="north"
            )
        )
        .add_trace(
            go.Scatter(
                x=df["date"], y=df["south"], mode="lines+markers", name="south"
            )
        )
        .update_layout(
            {
                "title": {"text": "seiryo's hemispheric sunspot number"},
                "xaxis": {"title": {"text": "date"}},
                "yaxis": {"title": {"text": "sunspot number"}},
            }
        )
    )


def main() -> None:
    path_seiryo = Path("out/seiryo/sn.parquet")
    output_path = Path("out/seiryo")

    df_seiryo = pl.read_parquet(path_seiryo).pipe(calc_sunspot_number)
    print(df_seiryo)
    df_seiryo.write_parquet(output_path / "sn_monthly.parquet")

    fig1 = draw_sunspot_number_whole_disk_plotly(df_seiryo)
    fig1.update_layout(
        {
            "template": "simple_white",
            "font_family": "Century",
            "title": {
                "font_size": 24,
                "x": 0.5,
                "y": 0.9,
                "xanchor": "center",
                "yanchor": "middle",
            },
            "xaxis": {
                "title_font_size": 20,
                "tickfont_size": 16,
                "linewidth": 1,
                "mirror": True,
                "showgrid": True,
                "ticks": "outside",
            },
            "yaxis": {
                "title_font_size": 20,
                "tickfont_size": 16,
                "linewidth": 1,
                "mirror": True,
                "showgrid": True,
                "ticks": "outside",
            },
        }
    )
    fig1.write_json(
        output_path / "sunspot_number_whole_disk.json", pretty=True
    )
    for ext in "pdf", "png":
        file_path = output_path / f"sunspot_number_whole_disk.{ext}"
        fig1.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )

    fig2 = draw_sunspot_number_hemispheric_plotly(df_seiryo)
    fig2.update_layout(
        {
            "template": "simple_white",
            "font_family": "Century",
            "title": {
                "font_size": 24,
                "x": 0.5,
                "y": 0.95,
                "xanchor": "center",
                "yanchor": "middle",
            },
            "legend": {
                "borderwidth": 1,
                "font_size": 16,
                "orientation": "h",
                "x": 0.5,
                "y": 1.1,
                "xanchor": "center",
                "yanchor": "middle",
            },
            "xaxis": {
                "title_font_size": 20,
                "tickfont_size": 16,
                "linewidth": 1,
                "mirror": True,
                "showgrid": True,
                "ticks": "outside",
            },
            "yaxis": {
                "title_font_size": 20,
                "tickfont_size": 16,
                "linewidth": 1,
                "mirror": True,
                "showgrid": True,
                "ticks": "outside",
            },
        }
    )
    fig2.write_json(
        output_path / "sunspot_number_hemispheric.json", pretty=True
    )
    for ext in "pdf", "png":
        file_path = output_path / f"sunspot_number_hemispheric.{ext}"
        fig2.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )


if __name__ == "__main__":
    main()
