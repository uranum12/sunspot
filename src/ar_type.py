from datetime import date
from enum import Enum, auto

import polars as pl


class SchemaType(Enum):
    NOTEBOOK_1 = auto()
    NOTEBOOK_2 = auto()
    NOTEBOOK_3 = auto()
    OLD = auto()
    NEW = auto()


def detect_schema_type(year: int, month: int) -> SchemaType | None:
    input_date = date(year, month, 1)

    schema_type_ranges: dict[SchemaType, tuple[date, date]] = {
        SchemaType.NOTEBOOK_1: (date(1953, 3, 1), date(1960, 12, 1)),
        SchemaType.NOTEBOOK_2: (date(1961, 1, 1), date(1963, 12, 1)),
        SchemaType.NOTEBOOK_3: (date(1964, 1, 1), date(1964, 3, 1)),
        SchemaType.OLD: (date(1964, 4, 1), date(1978, 1, 1)),
        SchemaType.NEW: (date(1978, 2, 1), date(2016, 6, 1)),
    }

    for schema_type, (start_date, end_date) in schema_type_ranges.items():
        if start_date <= input_date <= end_date:
            return schema_type

    return None


def detect_dtypes(schema_type: SchemaType) -> dict[str, type[pl.DataType]]:
    match schema_type:
        case SchemaType.NOTEBOOK_1:
            return {"no": pl.UInt32, "ns": pl.Categorical, "lat": pl.Utf8}
        case SchemaType.NOTEBOOK_2 | SchemaType.NOTEBOOK_3:
            return {"no": pl.Utf8, "ns": pl.Categorical, "lat": pl.Utf8}
        case SchemaType.OLD:
            return {
                "no": pl.Utf8,
                "lat": pl.Utf8,
                "lon": pl.Utf8,
                "first": pl.UInt8,
                "last": pl.UInt8,
            }
        case SchemaType.NEW:
            return {
                "no": pl.Utf8,
                "lat": pl.Utf8,
                "lon": pl.Utf8,
                "first": pl.Utf8,
                "last": pl.Utf8,
            }
