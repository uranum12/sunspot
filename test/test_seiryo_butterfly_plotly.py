from datetime import date

import numpy as np

import seiryo_butterfly
import seiryo_butterfly_plotly


def test_draw_butterfly_diagram() -> None:
    img = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]])
    info = seiryo_butterfly.ButterflyInfo(
        1,
        2,
        date(2020, 2, 1),
        date(2020, 4, 1),
        seiryo_butterfly.DateDelta(months=1),
    )
    _ = seiryo_butterfly_plotly.draw_butterfly_diagram_plotly(img, info)
