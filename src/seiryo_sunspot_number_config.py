from pydantic import BaseModel

from seiryo_config_common import Axis, FigSize, Legend, Line, Title


class SunspotNumberWholeDisk(BaseModel):
    fig_size: FigSize = FigSize()
    line: Line = Line()
    title: Title = Title(text="seiryo's whole-disk sunspot number")
    xaxis: Axis = Axis(title=Title(text="date"))
    yaxis: Axis = Axis(title=Title(text="sunspot number"))


class SunspotNumberHemispheric(BaseModel):
    fig_size: FigSize = FigSize()
    line_north: Line = Line(label="north")
    line_south: Line = Line(label="south")
    title: Title = Title(
        text="seiryo's hemispheric sunspot number", position=1.1
    )
    xaxis: Axis = Axis(title=Title(text="date"))
    yaxis: Axis = Axis(title=Title(text="sunspot number"))
    legend: Legend = Legend()
