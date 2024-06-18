from pydantic import BaseModel

from seiryo_config_common import Axis, Bar, FigSize, Title


class ObservationsDays(BaseModel):
    fig_size: FigSize = FigSize()
    bar: Bar = Bar(width=15)
    title: Title = Title(text="observations days per month")
    xaxis: Axis = Axis(title=Title(text="date"))
    yaxis: Axis = Axis(title=Title(text="observations days"))
