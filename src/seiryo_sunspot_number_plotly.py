from pathlib import Path

import plotly.graph_objects as go
import polars as pl


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
    data_path = Path("out/seiryo/sunspot/monthly.parquet")
    output_path = Path("out/seiryo/sunspot_plotly")
    output_path.mkdir(exist_ok=True)

    df = pl.read_parquet(data_path)

    fig_whole_disk = draw_sunspot_number_whole_disk_plotly(df)
    fig_whole_disk.update_layout(
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
    fig_whole_disk.write_json(
        output_path / "sunspot_number_whole_disk.json", pretty=True
    )
    for ext in "pdf", "png":
        file_path = output_path / f"sunspot_number_whole_disk.{ext}"
        fig_whole_disk.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )

    fig_hemispheric = draw_sunspot_number_hemispheric_plotly(df)
    fig_hemispheric.update_layout(
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
    fig_hemispheric.write_json(
        output_path / "sunspot_number_hemispheric.json", pretty=True
    )
    for ext in "pdf", "png":
        file_path = output_path / f"sunspot_number_hemispheric.{ext}"
        fig_hemispheric.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )


if __name__ == "__main__":
    main()
