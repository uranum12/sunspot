from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
import polars as pl
from matplotlib.figure import Figure


def calc_sunspot_number(df: pl.LazyFrame) -> pl.DataFrame:
    return (
        df.drop("time", "remarks")
        .drop_nulls()
        .with_columns(
            (pl.col("ng") + pl.col("sg")).alias("tg"),
            (pl.col("nf") + pl.col("sf")).alias("tf"),
        )
        .select(
            pl.col("date").dt.truncate("1mo"),
            # R = 10g + f
            (pl.col("ng").cast(pl.UInt16) * 10 + pl.col("nf")).alias("north"),
            (pl.col("sg").cast(pl.UInt16) * 10 + pl.col("sf")).alias("south"),
            (pl.col("tg").cast(pl.UInt16) * 10 + pl.col("tf")).alias("total"),
        )
        .group_by("date")
        .mean()
        .with_columns(
            ((pl.col("north") - pl.col("south")) / pl.col("total"))
            .fill_nan(0)
            .alias("index")
        )
        .sort("date")
        .collect()
    )


def draw_hemispheric(df: pl.DataFrame) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(df["date"], df["north"], lw=1, label="north")
    ax.plot(df["date"], df["south"], lw=1, label="south")

    ax.set_title("fujimori's sunspot number", y=1.1)
    ax.set_xlabel("year")
    ax.set_ylabel("relative sunspot number")

    ax.legend(
        fancybox=False,
        edgecolor="black",
        framealpha=1,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0,
        ncol=2,
    )

    return fig


def draw_hemispheric_plotly(df: pl.DataFrame) -> go.Figure:
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
                "title": {"text": "fujimori's hemispheric sunspot number"},
                "xaxis": {"title": {"text": "date"}},
                "yaxis": {"title": {"text": "sunspot number"}},
            }
        )
    )


def draw_asymmetry_index(df: pl.DataFrame) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(df["date"], df["index"], lw=1, label="index")

    ax.set_title("fujimori's hemispheric asymmetry index")
    ax.set_xlabel("date")
    ax.set_ylabel("asymmetry index")
    ax.grid()

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.tight_layout()

    return fig


def draw_asymmetry_index_plotly(df: pl.DataFrame) -> go.Figure:
    return (
        go.Figure()
        .add_trace(
            go.Scatter(x=df["date"], y=df["index"], mode="lines+markers")
        )
        .update_layout(
            {
                "title": {"text": "fujimori's hemispheric asymmetry index"},
                "xaxis": {"title": {"text": "date"}},
                "yaxis": {"title": {"text": "asymmetry index"}},
            }
        )
    )


def main_matplotlib() -> None:
    path_fujimori = Path("out/sn/all.parquet")
    output_path = Path("out/sn")

    df_fujimori = calc_sunspot_number(pl.scan_parquet(path_fujimori))
    print(df_fujimori)

    fig1 = draw_hemispheric(df_fujimori)

    for ext in "pdf", "png":
        fig1.savefig(
            output_path / f"hemispheric.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    fig2 = draw_asymmetry_index(df_fujimori)

    for ext in "pdf", "png":
        fig2.savefig(
            output_path / f"asymmetry_index.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )


def main_plotly() -> None:
    path_fujimori = Path("out/sn/all.parquet")
    output_path = Path("out/sn")

    df_fujimori = calc_sunspot_number(pl.scan_parquet(path_fujimori))
    print(df_fujimori)

    fig1 = draw_hemispheric_plotly(df_fujimori)
    fig1.update_layout(
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
    fig1.write_json(output_path / "hemispheric.json", pretty=True)
    for ext in "pdf", "png":
        file_path = output_path / f"hemispheric.{ext}"
        fig1.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )

    fig2 = draw_asymmetry_index_plotly(df_fujimori)
    fig2.update_layout(
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
                "tickformat": ".0%",
                "linewidth": 1,
                "mirror": True,
                "showgrid": True,
                "ticks": "outside",
            },
        }
    )
    fig2.write_json(output_path / "asymmetry_index.json", pretty=True)
    for ext in "pdf", "png":
        file_path = output_path / f"asymmetry_index.{ext}"
        fig2.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )


if __name__ == "__main__":
    main_plotly()
