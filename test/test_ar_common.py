import polars as pl
import pytest

import ar_common


@pytest.mark.parametrize(
    ("in_no", "out_ns", "out_no"),
    [
        ("N1", "N", "1"),
        ("N12", "N", "12"),
        ("N123", "N", "123"),
        ("S1234", "S", "1234"),
        ("S5432", "S", "5432"),
        ("N1234_56", "N", "1234_56"),
        ("S6543_2", "S", "6543_2"),
    ],
)
def test_extract_no(in_no: str, out_ns: str, out_no: str) -> None:
    df_in = pl.LazyFrame(
        {
            "no": [in_no],
        },
    )
    df_out = ar_common.extract_ns(df_in).collect()
    assert df_out.item(0, "ns") == out_ns
    assert df_out.item(0, "no") == out_no


@pytest.mark.parametrize(
    ("in_no", "out_no"),
    [
        ("1", 1),
        ("12", 12),
        ("123", 123),
        ("1234", 1234),
        ("12345", 12345),
        ("123456", 123456),
        ("400000", 400000),
    ],
)
def test_convert_no(in_no: str, out_no: int) -> None:
    df_in = pl.LazyFrame(
        {
            "no": [in_no],
        },
    )
    df_out = ar_common.convert_no(df_in).collect()
    assert df_out.item(0, "no") == out_no


@pytest.mark.parametrize(
    ("in_lat", "in_lon", "out_lat", "out_lon", "out_over"),
    [
        ("12", "123", "12", "123", False),
        ("12~15", "123~132", "12~15", "123~132", False),
        ("-2~3", "256", "-2~3", "256", False),
        ("/", "/", None, None, True),
    ],
)
def test_detect_coords_over(
    in_lat: str,
    in_lon: str,
    out_lat: str | None,
    out_lon: str | None,
    out_over: bool,
) -> None:
    df_in = pl.LazyFrame(
        {
            "lat": [in_lat],
            "lon": [in_lon],
        },
    )
    df_out = ar_common.detect_coords_over(df_in).collect()
    assert df_out.item(0, "lat") == out_lat
    assert df_out.item(0, "lon") == out_lon
    assert df_out.item(0, "over") == out_over


@pytest.mark.parametrize(
    (
        "in_lat",
        "in_lon",
        "out_lat",
        "out_lon",
        "out_lat_question",
        "out_lon_question",
    ),
    [
        ("12", "123", "12", "123", None, None),
        ("12?", "123", "12", "123", "?", None),
        ("12", "123?", "12", "123", None, "?"),
        ("12?", "123?", "12", "123", "?", "?"),
        ("12~15", "123", "12~15", "123", None, None),
        ("12~15?", "123", "12~15", "123", "?", None),
        ("12~15", "123?", "12~15", "123", None, "?"),
        ("12~15", "123~132", "12~15", "123~132", None, None),
        ("12~15", "123~132?", "12~15", "123~132", None, "?"),
    ],
)
def test_extract_coords_qm(
    in_lat: str,
    in_lon: str,
    out_lat: str,
    out_lon: str,
    out_lat_question: str | None,
    out_lon_question: str | None,
) -> None:
    df_in = pl.LazyFrame(
        {
            "lat": [in_lat],
            "lon": [in_lon],
        },
    )
    df_out = ar_common.extract_coords_qm(df_in).collect()
    assert df_out.item(0, "lat") == out_lat
    assert df_out.item(0, "lon") == out_lon
    assert df_out.item(0, "lat_question") == out_lat_question
    assert df_out.item(0, "lon_question") == out_lon_question


@pytest.mark.parametrize(
    ("in_lat", "out_lat", "out_lat_question"),
    [
        ("12", "12", None),
        ("12?", "12", "?"),
        ("-12", "-12", None),
        ("p12?", "p12", "?"),
        ("12~15", "12~15", None),
        ("12~15?", "12~15", "?"),
    ],
)
def test_extract_coords_qm_lat_only(
    in_lat: str,
    out_lat: str,
    out_lat_question: str | None,
) -> None:
    df_in = pl.LazyFrame(
        {
            "lat": [in_lat],
        },
    )
    df_out = ar_common.extract_coords_qm(df_in, ["lat"]).collect()
    assert df_out.item(0, "lat") == out_lat
    assert df_out.item(0, "lat_question") == out_lat_question


@pytest.mark.parametrize(
    (
        "in_lat",
        "in_lon",
        "out_lat_left",
        "out_lat_right",
        "out_lon_left",
        "out_lon_right",
    ),
    [
        ("12", "123", "12", "12", "123", "123"),
        ("12~15", "123", "12", "15", "123", "123"),
        ("12", "123~132", "12", "12", "123", "132"),
        ("-12", "123", "-12", "-12", "123", "123"),
        ("-12~15", "123", "-12", "15", "123", "123"),
        ("12~-15", "123", "12", "-15", "123", "123"),
        ("12~1", "123", "12", "1", "123", "123"),
        ("12", "123~121", "12", "12", "123", "121"),
        ("12~1", "123~121", "12", "1", "123", "121"),
    ],
)
def test_extract_coords_lr(
    in_lat: str,
    in_lon: str,
    out_lat_left: str,
    out_lat_right: str,
    out_lon_left: str,
    out_lon_right: str,
) -> None:
    df_in = pl.LazyFrame(
        {
            "lat": [in_lat],
            "lon": [in_lon],
        },
    )
    df_out = ar_common.extract_coords_lr(df_in).collect()
    assert df_out.item(0, "lat_left") == out_lat_left
    assert df_out.item(0, "lat_right") == out_lat_right
    assert df_out.item(0, "lon_left") == out_lon_left
    assert df_out.item(0, "lon_right") == out_lon_right


@pytest.mark.parametrize(
    ("in_lat", "out_lat_left", "out_lat_right"),
    [
        ("12", "12", "12"),
        ("12~15", "12", "15"),
        ("12~1", "12", "1"),
        ("-12~p15", "-12", "p15"),
    ],
)
def test_extract_coords_lr_lat_only(
    in_lat: str,
    out_lat_left: str,
    out_lat_right: str,
) -> None:
    df_in = pl.LazyFrame(
        {
            "lat": [in_lat],
        },
    )
    df_out = ar_common.extract_coords_lr(df_in, ["lat"]).collect()
    assert df_out.item(0, "lat_left") == out_lat_left
    assert df_out.item(0, "lat_right") == out_lat_right


@pytest.mark.parametrize(
    (
        "in_lat_left",
        "in_lat_right",
        "in_lon_left",
        "in_lon_right",
        "out_lat_left",
        "out_lat_right",
        "out_lon_left",
        "out_lon_right",
        "out_lat_left_sign",
        "out_lat_right_sign",
        "out_lon_left_sign",
        "out_lon_right_sign",
    ),
    [
        (
            "12",
            "12",
            "123",
            "123",
            "12",
            "12",
            "123",
            "123",
            None,
            None,
            None,
            None,
        ),
        (
            "-12",
            "p12",
            "-123",
            "p123",
            "12",
            "12",
            "123",
            "123",
            "-",
            "+",
            "-",
            "+",
        ),
    ],
)
def test_extract_coords_sign(
    in_lat_left: str,
    in_lat_right: str,
    in_lon_left: str,
    in_lon_right: str,
    out_lat_left: str,
    out_lat_right: str,
    out_lon_left: str,
    out_lon_right: str,
    out_lat_left_sign: str | None,
    out_lat_right_sign: str | None,
    out_lon_left_sign: str | None,
    out_lon_right_sign: str | None,
) -> None:
    df_in = pl.LazyFrame(
        {
            "lat_left": [in_lat_left],
            "lat_right": [in_lat_right],
            "lon_left": [in_lon_left],
            "lon_right": [in_lon_right],
        },
    )
    df_out = ar_common.extract_coords_sign(df_in).collect()
    assert df_out.item(0, "lat_left") == out_lat_left
    assert df_out.item(0, "lat_right") == out_lat_right
    assert df_out.item(0, "lon_left") == out_lon_left
    assert df_out.item(0, "lon_right") == out_lon_right
    assert df_out.item(0, "lat_left_sign") == out_lat_left_sign
    assert df_out.item(0, "lat_right_sign") == out_lat_right_sign
    assert df_out.item(0, "lon_left_sign") == out_lon_left_sign
    assert df_out.item(0, "lon_right_sign") == out_lon_right_sign


@pytest.mark.parametrize(
    (
        "in_lat_left",
        "in_lat_right",
        "out_lat_left",
        "out_lat_right",
        "out_lat_left_sign",
        "out_lat_right_sign",
    ),
    [
        ("12", "12", "12", "12", None, None),
        ("-12", "12", "12", "12", "-", None),
        ("-12", "p12", "12", "12", "-", "+"),
        ("p12", "-12", "12", "12", "+", "-"),
    ],
)
def test_extract_coords_sign_lat_only(
    in_lat_left: str,
    in_lat_right: str,
    out_lat_left: str,
    out_lat_right: str,
    out_lat_left_sign: str | None,
    out_lat_right_sign: str | None,
) -> None:
    df_in = pl.LazyFrame(
        {
            "lat_left": [in_lat_left],
            "lat_right": [in_lat_right],
        },
    )
    df_out = ar_common.extract_coords_sign(df_in, ["lat"]).collect()
    assert df_out.item(0, "lat_left") == out_lat_left
    assert df_out.item(0, "lat_right") == out_lat_right
    assert df_out.item(0, "lat_left_sign") == out_lat_left_sign
    assert df_out.item(0, "lat_right_sign") == out_lat_right_sign


@pytest.mark.parametrize(
    ("in_lat_left", "in_lat_right", "out_lat_left", "out_lat_right"),
    [
        ("12", "12", 12, 12),
        ("12", "15", 12, 15),
    ],
)
def test_convert_lat(
    in_lat_left: str,
    in_lat_right: str,
    out_lat_left: str,
    out_lat_right: str,
) -> None:
    df_in = pl.LazyFrame(
        {
            "lat_left": [in_lat_left],
            "lat_right": [in_lat_right],
        },
    )
    df_out = ar_common.convert_lat(df_in).collect()
    assert df_out.item(0, "lat_left") == out_lat_left
    assert df_out.item(0, "lat_right") == out_lat_right


@pytest.mark.parametrize(
    ("in_lon_left", "in_lon_right", "out_lon_left", "out_lon_right"),
    [
        ("123", "123", 123, 123),
        ("123", "132", 123, 132),
    ],
)
def test_convert_lon(
    in_lon_left: str,
    in_lon_right: str,
    out_lon_left: str,
    out_lon_right: str,
) -> None:
    df_in = pl.LazyFrame(
        {
            "lon_left": [in_lon_left],
            "lon_right": [in_lon_right],
        },
    )
    df_out = ar_common.convert_lon(df_in).collect()
    assert df_out.item(0, "lon_left") == out_lon_left
    assert df_out.item(0, "lon_right") == out_lon_right


def test_sort_by_row() -> None:
    in_rows = [
        "no",
        "ns",
        "over",
        "first",
        "last",
        "lat_left",
        "lat_right",
        "lon_left",
        "lon_right",
        "lat_left_sign",
        "lat_right_sign",
        "lon_left_sign",
        "lon_right_sign",
        "lat_question",
        "lon_question",
    ]
    out_rows = [
        "ns",
        "no",
        "lat_left",
        "lat_right",
        "lat_left_sign",
        "lat_right_sign",
        "lat_question",
        "lon_left",
        "lon_right",
        "lon_left_sign",
        "lon_right_sign",
        "lon_question",
        "first",
        "last",
        "over",
    ]
    df_in = pl.LazyFrame({row_name: [] for row_name in in_rows})
    df_out = ar_common.sort(df_in)
    assert df_out.columns == out_rows


@pytest.mark.parametrize(
    ("in_no", "in_ns", "out_no", "out_ns"),
    [
        (
            [1, 1, 3, 3, 2, 2],
            ["N", "S", "S", "N", "N", "S"],
            [1, 2, 3, 1, 2, 3],
            ["N", "N", "N", "S", "S", "S"],
        ),
        (
            [1, 2, None, 1, None, 3],
            [None, "N", "N", "S", "S", "S"],
            [1, None, 2, None, 1, 3],
            [None, "N", "N", "S", "S", "S"],
        ),
    ],
)
def test_sort_by_no(
    in_no: list[int],
    in_ns: list[str],
    out_no: list[int],
    out_ns: list[str],
) -> None:
    cols = [
        "lat_left",
        "lat_right",
        "lat_left_sign",
        "lat_right_sign",
        "lat_question",
        "lon_left",
        "lon_right",
        "lon_left_sign",
        "lon_right_sign",
        "lon_question",
        "first",
        "last",
        "over",
    ]
    df_in = pl.LazyFrame(
        {
            "no": in_no,
            "ns": in_ns,
        },
    ).with_columns([pl.lit(None).alias(col) for col in cols])
    df_out = ar_common.sort(df_in).collect()
    assert df_out.get_column("no").to_list() == out_no
    assert df_out.get_column("ns").to_list() == out_ns
