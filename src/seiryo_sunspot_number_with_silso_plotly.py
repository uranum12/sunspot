import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl


def draw_sunspot_number_with_silso_plotly(df: pl.DataFrame) -> go.Figure:
    return (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=df["date"],
                y=df["seiryo"],
                mode="lines+markers",
                name="seiryo",
            )
        )
        .add_trace(
            go.Scatter(
                x=df["date"], y=df["silso"], mode="lines+markers", name="silso"
            )
        )
        .update_layout(
            {
                "title": {"text": "seiryo's whole-disk sunspot number"},
                "xaxis": {"title": {"text": "date"}},
                "yaxis": {"title": {"text": "sunspot number"}},
            }
        )
    )


def draw_scatter_plotly(
    df: pl.DataFrame, factor: float, r2: float
) -> go.Figure:
    silso_max: float = df.select(pl.max("silso")).item()
    seiryo_max: float = df.select(pl.max("seiryo")).item()
    factor_max = max(silso_max, seiryo_max / factor)
    return (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=df["silso"],
                y=df["seiryo"],
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
                "title": {"text": "silso and seiryo"},
                "xaxis": {"title": {"text": "silso"}},
                "yaxis": {"title": {"text": "seiryo"}},
            }
        )
    )


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
                "title": {"text": "ratio: seiryo / silso"},
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
                "title": {"text": "difference: seiryo* - silso"},
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


def main() -> None:
    path_with_silso = Path("out/seiryo/sunspot/with_silso.parquet")
    path_ratio_diff = Path("out/seiryo/sunspot/ratio_diff.parquet")
    path_factor_r2 = Path("out/seiryo/sunspot/factor_r2.json")
    output_path = Path("out/seiryo/sunspot_plotly")
    output_path.mkdir(exist_ok=True)

    df_with_silso = pl.read_parquet(path_with_silso)
    df_ratio_diff = pl.read_parquet(path_ratio_diff)

    with path_factor_r2.open("r") as f:
        json_data = json.load(f)
        factor = json_data["factor"]
        r2 = json_data["r2"]

    fig_with_silso = draw_sunspot_number_with_silso_plotly(df_with_silso)
    fig_with_silso.update_layout(
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
    fig_with_silso.write_json(
        output_path / "sunspot_number_with_silso.json", pretty=True
    )
    for ext in "pdf", "png":
        file_path = output_path / f"sunspot_number_with_silso.{ext}"
        fig_with_silso.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )

    fig_scatter = draw_scatter_plotly(df_with_silso, factor, r2)
    fig_scatter.update_layout(
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
    fig_scatter.write_json(output_path / "scatter.json", pretty=True)
    for ext in "pdf", "png":
        file_path = output_path / f"scatter.{ext}"
        fig_scatter.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )

    fig_ratio = draw_ratio_plotly(df_ratio_diff)
    fig_ratio.update_layout(
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
    fig_ratio.write_json(output_path / "ratio.json", pretty=True)
    for ext in "pdf", "png":
        file_path = output_path / f"ratio.{ext}"
        fig_ratio.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )

    fig_diff = draw_diff_plotly(df_ratio_diff)
    fig_diff.update_layout(
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
    fig_diff.write_json(output_path / "diff.json", pretty=True)
    for ext in "pdf", "png":
        file_path = output_path / f"diff.{ext}"
        fig_diff.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )

    fig_ratio_diff = draw_ratio_and_diff_plotly(df_ratio_diff)
    fig_ratio_diff.update_layout(
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
    fig_ratio_diff.write_json(output_path / "ratio_and_diff.json", pretty=True)
    for ext in "pdf", "png":
        file_path = output_path / f"ratio_and_diff.{ext}"
        fig_ratio_diff.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )


if __name__ == "__main__":
    main()
