from pydantic import BaseModel, Field

from seiryo_config_common import Axis, FigSize, Image, Title


class Index(BaseModel):
    year_interval: int
    lat_interval: int


class ButterflyDiagram(BaseModel):
    fig_size: FigSize
    index: Index
    image: Image
    title: Title
    xaxis: Axis
    yaxis: Axis


class Color(BaseModel):
    red: int = Field(..., ge=0, le=0xFF)
    green: int = Field(..., ge=0, le=0xFF)
    blue: int = Field(..., ge=0, le=0xFF)


class ColorMap(BaseModel):
    cmap: list[Color]
