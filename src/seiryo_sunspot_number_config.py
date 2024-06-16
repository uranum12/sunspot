from pydantic import BaseModel


class FigSize(BaseModel):
    width: float = 8.0
    height: float = 5.0


class Line(BaseModel):
    label: str = ""
    style: str = "-"
    width: float = 1.0
    color: str | None = None


class Title(BaseModel):
    text: str
    font_family: str = "Times New Roman"
    font_size: int = 16
    position: float = 1.0


class Ticks(BaseModel):
    font_family: str = "Times New Roman"
    font_size: int = 12


class Axis(BaseModel):
    title: Title
    ticks: Ticks = Ticks()


class Legend(BaseModel):
    font_family: str = "Times New Roman"
    font_size: int = 12


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
