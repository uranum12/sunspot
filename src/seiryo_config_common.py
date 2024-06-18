from pydantic import BaseModel


class FigSize(BaseModel):
    width: float = 8.0
    height: float = 5.0


class Line(BaseModel):
    label: str = ""
    style: str = "-"
    width: float = 1.0
    color: str | None = None


class Bar(BaseModel):
    label: str = ""
    width: float = 1.0
    color: str | None = None


class Scatter(BaseModel):
    label: str = ""
    size: float = 10.0
    color: str | None = None
    edge_color: str | None = "none"


class Image(BaseModel):
    cmap: str = "binary"
    aspect: float = 1.0


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


class Text(BaseModel):
    x: float | None = None
    y: float | None = None
    math_font_family: str = "cm"
    font_family: str = "Times New Roman"
    font_size: int = 16


class Legend(BaseModel):
    font_family: str = "Times New Roman"
    font_size: int = 12
