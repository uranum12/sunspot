from pydantic import BaseModel

from seiryo_config_common import Axis, FigSize, Legend, Line, Title


class SunspotNumberWithFlare(BaseModel):
    fig_size: FigSize = FigSize()
    line_sunspot: Line = Line(color="C0", label="seiryo")
    line_flare: Line = Line(color="C1", label="flare")
    title: Title = Title(
        text="sunspot number and solar flare index", position=1.1
    )
    xaxis: Axis = Axis(title=Title(text="date"))
    yaxis_sunspot: Axis = Axis(title=Title(text="sunspot number"))
    yaxis_flare: Axis = Axis(title=Title(text="solar flare index"))
    legend: Legend = Legend()


class SunspotNumberWithFlareHemispheric(BaseModel):
    fig_size: FigSize = FigSize(height=8)
    title_north: Title = Title(text="North", position=1.1)
    title_south: Title = Title(text="South", position=1.1)
    xaxis: Axis = Axis(title=Title(text="date"))
    yaxis_north_sunspot: Axis = Axis(title=Title(text="sunspot number"))
    yaxis_north_flare: Axis = Axis(title=Title(text="solar flare index"))
    yaxis_south_sunspot: Axis = Axis(title=Title(text="sunspot number"))
    yaxis_south_flare: Axis = Axis(title=Title(text="solar flare index"))
    legend_north: Legend = Legend()
    legend_south: Legend = Legend()
