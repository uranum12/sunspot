import pytest

import sn_type


@pytest.mark.parametrize(
    ("in_year", "in_month", "out_time_type"),
    [
        (1955, 4, sn_type.TimeType.JST),
        (1959, 12, sn_type.TimeType.JST),
        (1960, 1, sn_type.TimeType.UT),
        (1960, 3, sn_type.TimeType.UT),
        (1960, 4, sn_type.TimeType.JST),
        (1968, 6, sn_type.TimeType.JST),
        (1968, 7, sn_type.TimeType.UT),
        (2020, 5, sn_type.TimeType.UT),
        (2020, 6, None),
    ],
)
def test_detect_time_type(
    in_year: int, in_month: int, out_time_type: sn_type.TimeType
) -> None:
    assert sn_type.detect_time_type(in_year, in_month) == out_time_type
