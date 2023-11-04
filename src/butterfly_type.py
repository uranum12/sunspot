from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DateDelta:
    years: int | None = None
    months: int | None = None
    days: int | None = None
