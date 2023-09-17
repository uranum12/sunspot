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
    in_year: int,
    in_month: int,
    out_schema_type: ar_type.SchemaType | None,
) -> None:
    assert ar_type.detect_schema_type(in_year, in_month) == out_schema_type
