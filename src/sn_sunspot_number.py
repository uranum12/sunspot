from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
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
        .sort("date")
        .collect()
    )


def join_data(
    df_fujimori: pl.DataFrame, df_silso: pl.DataFrame
) -> pl.DataFrame:
    df = (
        df_fujimori.lazy()
        .select("date", "total")
        .rename({"total": "fujimori"})
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
        lambda x, a: x * a, df["silso"], df["fujimori"]
    )
    return popt[0]


def calc_r2(df: pl.DataFrame, factor: float) -> float:
    r2 = metrics.r2_score(df["fujimori"], df["silso"] * factor)
    return float(r2)


def calc_ratio_and_diff(df: pl.DataFrame, factor: float) -> pl.DataFrame:
    return (
        df.lazy()
        .select(
            "date",
            (pl.col("fujimori") / pl.col("silso")).alias("ratio"),
            (pl.col("fujimori") / factor - pl.col("silso")).alias("diff"),
        )
        .sort("date")
        .collect()
    )


def draw_sunspot_number_whole_disk(df: pl.DataFrame) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(df["date"], df["fujimori"], lw=1, label="fujimori")
    ax.plot(df["date"], df["silso"], lw=1, label="SILSO")

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

    fig.tight_layout()

    return fig


def draw_sunspot_number_whole_disk_plotly(df: pl.DataFrame) -> go.Figure:
    return (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=df["date"],
                y=df["fujimori"],
                mode="lines+markers",
                name="fujimori",
            )
        )
        .add_trace(
            go.Scatter(
                x=df["date"], y=df["silso"], mode="lines+markers", name="silso"
            )
        )
        .update_layout(
            {
                "title": {"text": "fujimori's whole-disk sunspot number"},
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
                "title": {"text": "fujimori's hemispheric sunspot number"},
                "xaxis": {"title": {"text": "date"}},
                "yaxis": {"title": {"text": "sunspot number"}},
            }
        )
    )


def draw_scatter(df: pl.DataFrame, factor: float, r2: float) -> Figure:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(
        [0, 360], np.poly1d([factor, 0])([0, 360]), lw=1, c="black", zorder=1
    )
    ax.scatter(df["silso"], df["fujimori"], s=3, edgecolors="none", zorder=2)
    ax.text(300, 160, f"$y={factor:.5f}x$")
    ax.text(300, 140, f"$R^2={r2:.5f}$")

    ax.set_title("SILSO and fujimori")
    ax.set_xlabel("SILSO")
    ax.set_ylabel("fujimori")

    return fig


def draw_scatter_plotly(
    df: pl.DataFrame, factor: float, r2: float
) -> go.Figure:
    silso_max: float = df.select(pl.max("silso")).item()
    fujimori_max: float = df.select(pl.max("fujimori")).item()
    factor_max = max(silso_max, fujimori_max / factor)
    return (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=df["silso"],
                y=df["fujimori"],
                text=df["date"],
                mode="markers",
                name="sunspot number",
            )
        )
        .add_trace(
            go.Scatter(
                x=[0, factor_max],
                y=np.poly1d([factor, 0])([0, factor_max]),
                mode="lines",
                name="scaling factor",
                marker_color="black",
            )
        )
        .add_annotation(
            text=f"$y={factor:.5f}x$",
            showarrow=False,
            xref="paper",
            yref="paper",
            yanchor="top",
            x=0.8,
            y=0.2,
        )
        .add_annotation(
            text=f"$R^2={r2:.5f}$",
            showarrow=False,
            xref="paper",
            yref="paper",
            yanchor="bottom",
            x=0.8,
            y=0.2,
        )
        .update_layout(
            {
                "title": {"text": "silso and fujimori"},
                "xaxis": {"title": {"text": "silso"}},
                "yaxis": {"title": {"text": "fujimori"}},
            }
        )
    )


def draw_ratio_and_diff(df: pl.DataFrame, factor: float) -> Figure:
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.axhline(y=factor, c="black", lw=1, ls="--", zorder=1)
    ax1.plot(df["date"], df["ratio"], lw=1, zorder=2)

    ax1.set_title("ratio: fujimori / SILSO")
    ax1.set_xlabel("year")
    ax1.set_ylabel("ratio")
    ax1.grid()

    ax2.plot(df["date"], df["diff"], lw=1)

    ax2.set_title("difference: fujimori* - SILSO")
    ax2.set_xlabel("year")
    ax2.set_ylabel("difference")
    ax2.grid()

    fig.tight_layout()

    return fig


def draw_ratio_plotly(df: pl.DataFrame) -> go.Figure:
    return (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=df["date"], y=df["ratio"], mode="lines+markers", name="ratio"
            )
        )
        .update_layout(
            {
                "title": {"text": "ratio: fujimori / silso"},
                "xaxis": {"title": {"text": "date"}},
                "yaxis": {"title": {"text": "ratio"}},
            }
        )
    )


def draw_diff_plotly(df: pl.DataFrame) -> go.Figure:
    return (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=df["date"], y=df["diff"], mode="lines+markers", name="diff"
            )
        )
        .update_layout(
            {
                "title": {"text": "difference: fujimori* - silso"},
                "xaxis": {"title": {"text": "date"}},
                "yaxis": {"title": {"text": "difference"}},
            }
        )
    )


def draw_ratio_and_diff_plotly(df: pl.DataFrame) -> go.Figure:
    return (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=df["date"], y=df["ratio"], mode="lines+markers", name="ratio"
            )
        )
        .add_trace(
            go.Scatter(
                x=df["date"],
                y=df["diff"],
                yaxis="y2",
                mode="lines+markers",
                name="diff",
            )
        )
        .update_layout(
            {
                "title": {"text": "ratio and difference"},
                "xaxis": {"title": {"text": "date"}},
                "yaxis": {"title": {"text": "ratio"}, "side": "left"},
                "yaxis2": {
                    "title": {"text": "difference"},
                    "overlaying": "y",
                    "side": "right",
                },
            }
        )
    )


def main_matplotlib() -> None:
    path_fujimori = Path("out/sn/all.parquet")
    path_silso = Path("data/SN_m_tot_V2.0.txt")
    output_path = Path("out/sn")

    df_fujimori = calc_sunspot_number(pl.scan_parquet(path_fujimori))
    print(df_fujimori)

    df_silso = load_silso_data(path_silso)
    print(df_silso)

    df_joined = join_data(df_fujimori, df_silso)
    print(df_joined)

    fig1 = draw_sunspot_number_whole_disk(df_joined)

    for ext in "pdf", "png":
        fig1.savefig(
            output_path / f"sunspot_number.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    fig2 = draw_sunspot_number_hemispheric(df_fujimori)

    for ext in "pdf", "png":
        fig2.savefig(
            output_path / f"sunspot_number_sn.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    factor = calc_factor(df_joined)
    print(f"{factor=}")

    r2 = calc_r2(df_joined, factor)
    print(f"{r2=}")

    fig3 = draw_scatter(df_joined, factor, r2)

    for ext in "pdf", "png":
        fig3.savefig(
            output_path / f"scatter.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    df_ratio_diff = calc_ratio_and_diff(df_joined, factor)
    print(df_ratio_diff)

    fig4 = draw_ratio_and_diff(df_ratio_diff, factor)

    for ext in "pdf", "png":
        fig4.savefig(
            output_path / f"ratio_diff.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    plt.show()


def main_plotly() -> None:
    path_fujimori = Path("out/sn/all.parquet")
    path_silso = Path("data/SN_m_tot_V2.0.txt")
    output_path = Path("out/sn")

    df_fujimori = calc_sunspot_number(pl.scan_parquet(path_fujimori))
    print(df_fujimori)

    df_silso = load_silso_data(path_silso)
    print(df_silso)

    df_joined = join_data(df_fujimori, df_silso)
    print(df_joined)

    fig1 = draw_sunspot_number_whole_disk_plotly(df_joined)
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
    fig1.write_json(
        output_path / "sunspot_number_whole_disk.json", pretty=True
    )
    for ext in "pdf", "png":
        file_path = output_path / f"sunspot_number_whole_disk.{ext}"
        fig1.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )

    fig2 = draw_sunspot_number_hemispheric_plotly(df_fujimori)
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

    factor = calc_factor(df_joined)
    print(f"{factor=}")

    r2 = calc_r2(df_joined, factor)
    print(f"{r2=}")

    fig3 = draw_scatter_plotly(df_joined, factor, r2)
    fig3.update_layout(
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
    fig3.write_json(output_path / "scatter.json", pretty=True)
    for ext in "pdf", "png":
        file_path = output_path / f"scatter.{ext}"
        fig3.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )

    df_ratio_diff = calc_ratio_and_diff(df_joined, factor)
    print(df_ratio_diff)

    fig4 = draw_ratio_plotly(df_ratio_diff)
    fig4.update_layout(
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
    fig4.write_json(output_path / "ratio.json", pretty=True)
    for ext in "pdf", "png":
        file_path = output_path / f"ratio.{ext}"
        fig4.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )

    fig5 = draw_diff_plotly(df_ratio_diff)
    fig5.update_layout(
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
    fig5.write_json(output_path / "diff.json", pretty=True)
    for ext in "pdf", "png":
        file_path = output_path / f"diff.{ext}"
        fig5.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )

    fig6 = draw_ratio_and_diff_plotly(df_ratio_diff)
    fig6.update_layout(
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
                "showgrid": True,
                "ticks": "outside",
            },
            "yaxis2": {
                "title_font_size": 20,
                "tickfont_size": 16,
                "linewidth": 1,
                "showgrid": True,
                "ticks": "outside",
            },
        }
    )
    fig6.write_json(output_path / "ratio_and_diff.json", pretty=True)
    for ext in "pdf", "png":
        file_path = output_path / f"ratio_and_diff.{ext}"
        fig6.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )


if __name__ == "__main__":
    main_plotly()
