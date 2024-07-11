import json
from pathlib import Path
from pprint import pprint

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

import seiryo_butterfly_draw
from seiryo_butterfly import ButterflyInfo


def draw_butterfly_diagram_plotly(
    img: npt.NDArray[np.uint8], info: ButterflyInfo
) -> go.Figure:
    date_index = seiryo_butterfly_draw.create_date_index(
        info.date_start, info.date_end, info.date_interval.to_interval()
    )
    lat_index = seiryo_butterfly_draw.create_lat_index(
        info.lat_min, info.lat_max
    )

    xlabel = [
        (i, f"{d.year}")
        for i, d in enumerate(item.item() for item in date_index)
        if d.month == 1 and d.year % 2 == 0
    ]
    ylabel = [(i, n) for i, n in enumerate(lat_index) if n % 10 == 0]
    return (
        go.Figure()
        .add_trace(
            go.Heatmap(
                z=img, showscale=False, colorscale=[[0, "white"], [1, "black"]]
            )
        )
        .update_layout(
            {
                "title": {"text": "butterfly diagram"},
                "xaxis": {
                    "title": {"text": "date"},
                    "constrain": "domain",
                    "tickmode": "array",
                    "tickvals": [i[0] for i in xlabel],
                    "ticktext": [i[1] for i in xlabel],
                },
                "yaxis": {
                    "title": {"text": "latitude"},
                    "autorange": "reversed",
                    "scaleanchor": "x",
                    "constrain": "domain",
                    "tickmode": "array",
                    "tickvals": [i[0] for i in ylabel],
                    "ticktext": [i[1] for i in ylabel],
                },
            }
        )
    )


def main() -> None:
    data_path = Path("out/seiryo/butterfly/monthly.npz")
    info_path = Path("out/seiryo/butterfly/monthly.json")
    output_path = Path("out/seiryo/butterfly_plotly")
    output_path.mkdir(parents=True, exist_ok=True)

    with info_path.open("r") as f_info:
        info = ButterflyInfo.from_dict(json.load(f_info))
    pprint(info)

    with np.load(data_path) as f_img:
        img = f_img["img"]
    print(img)

    fig = draw_butterfly_diagram_plotly(img, info)
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
                "ticks": "outside",
            },
            "yaxis": {
                "title_font_size": 20,
                "tickfont_size": 16,
                "linewidth": 1,
                "mirror": True,
                "ticks": "outside",
            },
        }
    )

    fig.write_json(output_path / "butterfly_diagram.json", pretty=True)
    for ext in "pdf", "png":
        file_path = output_path / f"butterfly_diagram.{ext}"
        fig.write_image(
            file_path, width=800, height=500, engine="kaleido", scale=10
        )


if __name__ == "__main__":
    main()
