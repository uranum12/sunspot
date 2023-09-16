from datetime import date
from enum import Enum, auto


class TimeType(Enum):
    JST = auto()
    UT = auto()


def detect_time_type(year: int, month: int) -> TimeType | None:
    input_date = date(year, month, 1)

    time_type_ranges: dict[TimeType, list[tuple[date, date]]] = {
        TimeType.JST: [
            (date(1954, 1, 1), date(1959, 12, 1)),
            (date(1960, 4, 1), date(1968, 6, 1)),
        ],
        TimeType.UT: [
            (date(1960, 1, 1), date(1960, 3, 1)),
            (date(1968, 7, 1), date(2020, 5, 1)),
        ],
    }

    for time_type, date_ranges in time_type_ranges.items():
        for start_date, end_date in date_ranges:
            if start_date <= input_date <= end_date:
                return time_type

    return None
