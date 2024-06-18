from pydantic import BaseModel

from seiryo_config_common import Axis, FigSize, Legend, Line, Text, Title
from seiryo_config_common import Scatter as ScatterPlot


class SunspotNumberWithSilso(BaseModel):
    fig_size: FigSize = FigSize()
    line_seiryo: Line = Line(label="seiryo")
    line_silso: Line = Line(label="SILSO")
    title: Title = Title(
        text="seiryo's whole-disk sunspot number", position=1.1
    )
    xaxis: Axis = Axis(title=Title(text="date"))
    yaxis: Axis = Axis(title=Title(text="sunspot number"))
    legend: Legend = Legend()


class Scatter(BaseModel):
    fig_size: FigSize = FigSize()
    line_factor: Line = Line(color="black")
    scatter: ScatterPlot = ScatterPlot()
    text_factor: Text = Text()
    text_r2: Text = Text()
    title: Title = Title(text="SILSO and seiryo")
    xaxis: Axis = Axis(title=Title(text="SILSO"))
    yaxis: Axis = Axis(title=Title(text="seiryo"))


class Ratio(BaseModel):
    fig_size: FigSize = FigSize()
    line_factor: Line = Line(style="--", color="black")
    line_ratio: Line = Line()
    title: Title = Title(text="ratio: seiryo / SILSO")
    xaxis: Axis = Axis(title=Title(text="date"))
    yaxis: Axis = Axis(title=Title(text="ratio"))


class Diff(BaseModel):
    fig_size: FigSize = FigSize()
    line: Line = Line()
    title: Title = Title(text="difference: seiryo* - SILSO")
    xaxis: Axis = Axis(title=Title(text="date"))
    yaxis: Axis = Axis(title=Title(text="difference"))


class RatioDiff1(BaseModel):
    fig_size: FigSize = FigSize(height=8)
    line_factor: Line = Line(style="--", color="black")
    line_ratio: Line = Line()
    line_diff: Line = Line()
    title_ratio: Title = Title(text="ratio: seiryo / SILSO")
    title_diff: Title = Title(text="difference: seiryo* - SILSO")
    xaxis: Axis = Axis(title=Title(text="date"))
    yaxis_ratio: Axis = Axis(title=Title(text="ratio"))
    yaxis_diff: Axis = Axis(title=Title(text="difference"))


class RatioDiff2(BaseModel):
    fig_size: FigSize = FigSize()
    line_ratio: Line = Line(color="C0", label="ratio")
    line_diff: Line = Line(color="C1", label="diff")
    title: Title = Title(
        text="ratio: seiryo / SILSO and difference: seiryo* - SILSO",
        position=1.1,
    )
    xaxis: Axis = Axis(title=Title(text="date"))
    yaxis_ratio: Axis = Axis(title=Title(text="ratio"))
    yaxis_diff: Axis = Axis(title=Title(text="difference"))
    legend: Legend = Legend()
