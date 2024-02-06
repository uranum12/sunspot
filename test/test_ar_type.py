import polars as pl
import pytest

import ar_type


@pytest.mark.parametrize(
    ("in_year", "in_month", "out_schema_type"),
    [
        (1954, 1, ar_type.SchemaType.NOTEBOOK_1),
        (1960, 12, ar_type.SchemaType.NOTEBOOK_1),
        (1961, 1, ar_type.SchemaType.NOTEBOOK_2),
        (1963, 12, ar_type.SchemaType.NOTEBOOK_2),
        (1964, 1, ar_type.SchemaType.NOTEBOOK_3),
        (1964, 3, ar_type.SchemaType.NOTEBOOK_3),
        (1964, 4, ar_type.SchemaType.OLD),
        (1978, 1, ar_type.SchemaType.OLD),
        (1978, 2, ar_type.SchemaType.NEW),
        (2016, 6, ar_type.SchemaType.NEW),
        (2016, 7, None),
    ],
)
def test_detect_schema_type(
    in_year: int, in_month: int, out_schema_type: ar_type.SchemaType | None
) -> None:
    assert ar_type.detect_schema_type(in_year, in_month) == out_schema_type


@pytest.mark.parametrize(
    ("in_schema_type", "out_dtypes"),
    [
        (
            ar_type.SchemaType.NOTEBOOK_1,
            {"no": pl.UInt32, "ns": pl.Categorical, "lat": pl.Utf8},
        ),
        (
            ar_type.SchemaType.NOTEBOOK_2,
            {"no": pl.Utf8, "ns": pl.Categorical, "lat": pl.Utf8},
        ),
        (
            ar_type.SchemaType.NOTEBOOK_3,
            {"no": pl.Utf8, "ns": pl.Categorical, "lat": pl.Utf8},
        ),
        (
            ar_type.SchemaType.OLD,
            {
                "no": pl.Utf8,
                "lat": pl.Utf8,
                "lon": pl.Utf8,
                "first": pl.UInt8,
                "last": pl.UInt8,
            },
        ),
        (
            ar_type.SchemaType.NEW,
            {
                "no": pl.Utf8,
                "lat": pl.Utf8,
                "lon": pl.Utf8,
                "first": pl.Utf8,
                "last": pl.Utf8,
            },
        ),
    ],
)
def test_detect_dtypes(
    in_schema_type: ar_type.SchemaType,
    out_dtypes: dict[str, pl.PolarsDataType],
) -> None:
    assert ar_type.detect_dtypes(in_schema_type) == out_dtypes
