from pydantic import BaseModel

from seiryo_config_common import Axis, Bar, FigSize, Title


class ObservationsMonthly(BaseModel):
    fig_size: FigSize
    bar: Bar
    title: Title
    xaxis: Axis
    yaxis: Axis
