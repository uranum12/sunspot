from pathlib import Path

import plotly.graph_objects as go
import polars as pl


def draw_monthly_obs_days_plotly(df: pl.DataFrame) -> go.Figure:
    return (
        go.Figure()
        .add_trace(go.Bar(x=df["date"], y=df["obs"]))
        .update_layout(
            {
                "title": {"text": "observations days per month"},
                "xaxis": {"title": {"text": "date"}},
                "yaxis": {"title": {"text": "observations days"}},
            }
        )
    )


def main() -> None:
    data_file = Path("out/seiryo/observations/monthly.parquet")
    output_path = Path("out/seiryo/observations_plotly")
    output_path.mkdir(exist_ok=True)

    df = pl.read_parquet(data_file)

    fig = draw_monthly_obs_days_plotly(df)
    fig.update_layout(
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
    fig.write_json(output_path / "monthly.json", pretty=True)
    for ext in "pdf", "png":
        file_path = output_path / f"monthly.{ext}"
        fig.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )


if __name__ == "__main__":
    main()
