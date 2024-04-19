from pathlib import Path

import plotly.graph_objects as go
import polars as pl


def draw_sunspot_number_with_flare_plotly(df: pl.DataFrame) -> go.Figure:
    return (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=df["date"],
                y=df["seiryo"],
                mode="lines+markers",
                name="Seiryo",
            )
        )
        .add_trace(
            go.Scatter(
                x=df["date"],
                y=df["flare"],
                yaxis="y2",
                mode="lines+markers",
                name="Flare",
            )
        )
        .update_layout(
            {
                "title": "sunspot number and solar flare index",
                "xaxis": {"title": {"text": "date"}},
                "yaxis": {"title": {"text": "sunspot number"}, "side": "left"},
                "yaxis2": {
                    "title": {"text": "solar flare index"},
                    "overlaying": "y",
                    "side": "right",
                },
            }
        )
    )


def main() -> None:
    path_with_flare = Path("out/seiryo/sunspot/with_flare.parquet")
    output_path = Path("out/seiryo/sunspot_plotly")

    df_with_flare = pl.read_parquet(path_with_flare)

    fig_with_flare = draw_sunspot_number_with_flare_plotly(df_with_flare)
    fig_with_flare.update_layout(
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
    fig_with_flare.write_json(
        output_path / "sunspot_number_with_flare.json", pretty=True
    )
    for ext in "pdf", "png":
        file_path = output_path / f"sunspot_number_with_flare.{ext}"
        fig_with_flare.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )


if __name__ == "__main__":
    main()
