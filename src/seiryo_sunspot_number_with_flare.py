from datetime import date
from pathlib import Path

import plotly.graph_objects as go
import polars as pl


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
    path_sn = Path("out/seiryo/sn_monthly.parquet")
    path_flare = Path("data/flare")
    output_path = Path("out/seiryo")

    df_sn = pl.read_parquet(path_sn)
    print(df_sn)

    df_flare = load_flare_data(path_flare)
    print(df_flare)

    df_joined = join_data(df_sn, df_flare)
    print(df_joined)

    fig = draw_sunspot_number_with_flare_plotly(df_joined)
    fig.update_layout(
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
    fig.write_json(output_path / "sunspot_number_with_flare.json", pretty=True)
    for ext in "pdf", "png":
        file_path = output_path / f"sunspot_number_with_flare.{ext}"
        fig.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )


if __name__ == "__main__":
    main()
