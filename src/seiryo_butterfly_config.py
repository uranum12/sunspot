from pydantic import BaseModel

from seiryo_config_common import Axis, FigSize, Image, Title


class Index(BaseModel):
    year_inteerval: int = 10
    lat_interval: int = 10


class ButterflyDiagram(BaseModel):
    fig_size: FigSize = FigSize()
    index: Index = Index()
    image: Image = Image()
    title: Title = Title(text="butterfly diagram")
    xaxis: Axis = Axis(title=Title(text="date"))
    yaxis: Axis = Axis(title=Title(text="latitude"))
