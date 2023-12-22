from datetime import date
from random import sample

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import seiryo_agg


@pytest.mark.parametrize(
    ("in_date", "out_date"),
    [
        (
            [
                date(2020, 8, 10),
                date(2020, 8, 20),
                None,
            ],
            [
                date(2020, 8, 10),
                date(2020, 8, 20),
                date(2020, 8, 20),
            ],
        ),
        (
            [
                date(1965, 3, 20),
                None,
                None,
                date(1970, 1, 1),
                None,
            ],
            [
                date(1965, 3, 20),
                date(1965, 3, 20),
                date(1965, 3, 20),
                date(1970, 1, 1),
                date(1970, 1, 1),
            ],
        ),
    ],
)
def test_fill_date(in_date: list[date | None], out_date: list[date]) -> None:
    df_in = pl.LazyFrame(
        {
            "date": in_date,
        },
        schema={
            "date": pl.Date,
        },
    )
    df_expected = pl.LazyFrame(
        {
            "date": out_date,
        },
        schema={
            "date": pl.Date,
        },
    )
    df_out = seiryo_agg.fill_date(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
    )


@pytest.mark.parametrize(
    ("in_no", "in_num", "out_no", "out_num"),
    [
        ("1", "5", 1, 5),
        ("2", "1", 2, 1),
        ("3", "25", 3, 25),
        ("0", None, 0, None),
    ],
)
def test_convert_number(
    in_no: str,
    in_num: str | None,
    out_no: int,
    out_num: int | None,
) -> None:
    df_in = pl.LazyFrame(
        {
            "no": [in_no],
            "num": [in_num],
        },
        schema={
            "no": pl.Utf8,
            "num": pl.Utf8,
        },
    )
    df_expected = pl.LazyFrame(
        {
            "no": [out_no],
            "num": [out_num],
        },
        schema={
            "no": pl.UInt8,
            "num": pl.UInt16,
        },
    )
    df_out = seiryo_agg.convert_number(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
    )


@pytest.mark.parametrize(
    ("in_date", "out_date"),
    [
        ("2020/8/20", date(2020, 8, 20)),
        ("2020-8-20", date(2020, 8, 20)),
        ("2020.8.20", date(2020, 8, 20)),
        ("2020 8 20", date(2020, 8, 20)),
        ("2020/08/20", date(2020, 8, 20)),
        ("2020-08-20", date(2020, 8, 20)),
        ("2020.08.20", date(2020, 8, 20)),
        ("2020 08 20", date(2020, 8, 20)),
        ("1970-01-01", date(1970, 1, 1)),
        ("1965-12-21", date(1965, 12, 21)),
    ],
)
def test_convert_date(in_date: str, out_date: date) -> None:
    df_in = pl.LazyFrame(
        {
            "date": [in_date],
        },
        schema={
            "date": pl.Utf8,
        },
    )
    df_expected = pl.LazyFrame(
        {
            "date": [out_date],
        },
        schema={
            "date": pl.Date,
        },
    )
    df_out = seiryo_agg.convert_date(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
    )


@pytest.mark.parametrize(
    (
        "in_lat",
        "in_lon",
        "out_lat_min",
        "out_lat_max",
        "out_lon_min",
        "out_lon_max",
    ),
    [
        ("12", "15", 12, 12, 15, 15),
        ("+12", "+15", 12, 12, 15, 15),
        ("-12", "-15", -12, -12, -15, -15),
        ("1~2", "3~4", 1, 2, 3, 4),
        ("2~1", "4~3", 1, 2, 3, 4),
        ("-2~-1", "-20~-30", -2, -1, -30, -20),
        ("-1~-2", "-30~-20", -2, -1, -30, -20),
        ("1~-2", "+3~4", -2, 1, 3, 4),
        ("-2~1", "3~4", -2, 1, 3, 4),
        ("+1~-2", "3~+4", -2, 1, 3, 4),
        ("p12", "p15", 12, 12, 15, 15),
        ("m12", "m15", -12, -12, -15, -15),
        ("P12", "P15", 12, 12, 15, 15),
        ("M12", "M15", -12, -12, -15, -15),
        ("p1~p2", "p3~4", 1, 2, 3, 4),
        ("m2~m1", "m20~m30", -2, -1, -30, -20),
        ("m1~m2", "m30~m20", -2, -1, -30, -20),
        ("N3", "W30", 3, 3, -30, -30),
        ("S3", "E30", -3, -3, 30, 30),
        ("N3~4", "E10~12", 3, 4, 10, 12),
        ("N4~3", "E12~10", 3, 4, 10, 12),
        ("N3~N4", "E10~E12", 3, 4, 10, 12),
        ("S3~4", "W10~12", -4, -3, -12, -10),
        ("S4~3", "W12~10", -4, -3, -12, -10),
        ("S3~S4", "W10~W12", -4, -3, -12, -10),
        ("N1~S2", "E3~4", -2, 1, 3, 4),
        ("n3", "w30", 3, 3, -30, -30),
        ("s3", "e30", -3, -3, 30, 30),
        ("n3~4", "e10~12", 3, 4, 10, 12),
        ("n4~3", "e12~10", 3, 4, 10, 12),
        ("n3~n4", "e10~e12", 3, 4, 10, 12),
        ("s3~4", "w10~12", -4, -3, -12, -10),
        ("s4~3", "w12~10", -4, -3, -12, -10),
        ("s3~s4", "w10~w12", -4, -3, -12, -10),
        ("n1~s2", "e3~4", -2, 1, 3, 4),
        ("0.5", "1.2", 1, 1, 1, 1),
        ("1.5", "2.2", 2, 2, 2, 2),
        ("2.5", "3.2", 3, 3, 3, 3),
        ("3.5", "4.2", 4, 4, 4, 4),
        ("4.5", "5.2", 5, 5, 5, 5),
        ("5.5", "6.2", 6, 6, 6, 6),
        ("6.5", "7.2", 7, 7, 7, 7),
        ("7.5", "8.2", 8, 8, 8, 8),
        ("8.5", "9.2", 9, 9, 9, 9),
        ("9.5", "10.2", 10, 10, 10, 10),
        ("-0.5", "-1.2", -1, -1, -1, -1),
        ("-1.5", "-2.2", -2, -2, -2, -2),
        ("-2.5", "-3.2", -3, -3, -3, -3),
        ("-3.5", "-4.2", -4, -4, -4, -4),
        ("-4.5", "-5.2", -5, -5, -5, -5),
        ("m5.5", "m6.2", -6, -6, -6, -6),
        ("m6.5", "m7.2", -7, -7, -7, -7),
        ("m7.5", "m8.2", -8, -8, -8, -8),
        ("m8.5", "m9.2", -9, -9, -9, -9),
        ("m9.5", "m10.2", -10, -10, -10, -10),
        ("0.3~0.2", "-0.3~-0.2", 0, 0, 0, 0),
        ("1.5~2.5", "12.5~13.5", 2, 3, 13, 14),
        ("N3.7~5.5", "W4.1~6.5", 4, 6, -7, -4),
    ],
)
def test_convert_lat(
    in_lat: str,
    in_lon: str,
    out_lat_min: int,
    out_lat_max: int,
    out_lon_min: int,
    out_lon_max: int,
) -> None:
    df_in = pl.LazyFrame(
        {
            "lat": [in_lat],
            "lon": [in_lon],
        },
        schema={
            "lat": pl.Utf8,
            "lon": pl.Utf8,
        },
    )
    df_expected = pl.LazyFrame(
        {
            "lat_min": [out_lat_min],
            "lat_max": [out_lat_max],
            "lon_min": [out_lon_min],
            "lon_max": [out_lon_max],
        },
        schema={
            "lat_min": pl.Int8,
            "lat_max": pl.Int8,
            "lon_min": pl.Int16,
            "lon_max": pl.Int16,
        },
    )
    df_out = seiryo_agg.convert_lat(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
    )


@pytest.mark.parametrize(
    ("in_no", "out_no_spot", "out_no_nospot"),
    [
        ([0, 1, 2, 3, 4], [1, 2, 3, 4], [0]),
        ([1, 2, 3], [1, 2, 3], []),
        ([0, 1, 0, 1, 0, 0, 1, 2, 3], [1, 1, 1, 2, 3], [0, 0, 0, 0]),
        ([0, 0, 0, 0], [], [0, 0, 0, 0]),
    ],
)
def test_split(
    in_no: list[int],
    out_no_spot: list[int],
    out_no_nospot: list[int],
) -> None:
    df_in = pl.LazyFrame(
        {
            "no": in_no,
        },
        schema={
            "no": pl.UInt8,
        },
    )
    df_expected_spot = pl.LazyFrame(
        {
            "no": out_no_spot,
        },
        schema={
            "no": pl.UInt8,
        },
    )
    df_expected_nospot = pl.LazyFrame(
        {
            "no": out_no_nospot,
        },
        schema={
            "no": pl.UInt8,
        },
    )
    df_out_spot, df_out_nospot = seiryo_agg.split(df_in)
    assert_frame_equal(
        df_out_spot,
        df_expected_spot,
        check_column_order=False,
    )
    assert_frame_equal(
        df_out_nospot,
        df_expected_nospot,
        check_column_order=False,
    )


@pytest.mark.parametrize(
    ("in_lat_min", "in_lat_max", "out_lat"),
    [
        (8, 8, "N"),
        (-3, -3, "S"),
        (2, 3, "N"),
        (-1, 5, "N"),
        (-2, 2, "N"),
        (-3, -1, "S"),
        (-3, 1, "S"),
        (0, 0, "N"),
    ],
)
def test_calc_lat(in_lat_min: int, in_lat_max: int, out_lat: str) -> None:
    df_in = pl.LazyFrame(
        {
            "lat_min": [in_lat_min],
            "lat_max": [in_lat_max],
        },
        schema={
            "lat_min": pl.Int8,
            "lat_max": pl.Int8,
        },
    )
    df_expected = pl.LazyFrame(
        {
            "lat": [out_lat],
        },
        schema={
            "lat": pl.Utf8,
        },
    )
    df_out = seiryo_agg.calc_lat(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
    )


@pytest.mark.parametrize(
    (
        "in_date",
        "in_lat",
        "in_num",
        "out_date",
        "out_ng",
        "out_nf",
        "out_sg",
        "out_sf",
        "out_tg",
        "out_tf",
    ),
    [
        (
            [date(2020, 8, 1), date(2020, 8, 1), date(2020, 8, 1)],
            ["N", "N", "S"],
            [5, 6, 8],
            [date(2020, 8, 1)],
            [2],
            [11],
            [1],
            [8],
            [3],
            [19],
        ),
        (
            [date(1965, 4, 5), date(1965, 4, 5), date(1965, 4, 5)],
            ["N", "S", "S"],
            [4, 8, 12],
            [date(1965, 4, 5)],
            [1],
            [4],
            [2],
            [20],
            [3],
            [24],
        ),
        (
            [date(2000, 12, 24), date(2000, 12, 24), date(2000, 12, 25)],
            ["S", "S", "S"],
            [1, 2, 2],
            [date(2000, 12, 24), date(2000, 12, 25)],
            [0, 0],
            [0, 0],
            [2, 1],
            [3, 2],
            [2, 1],
            [3, 2],
        ),
    ],
)
def test_calc_sn(
    in_date: list[date],
    in_lat: list[str],
    in_num: list[int],
    out_date: list[date],
    out_ng: list[int],
    out_nf: list[int],
    out_sg: list[int],
    out_sf: list[int],
    out_tg: list[int],
    out_tf: list[int],
) -> None:
    df_in = pl.LazyFrame(
        {
            "date": in_date,
            "lat": in_lat,
            "num": in_num,
        },
        schema={
            "date": pl.Date,
            "lat": pl.Utf8,
            "num": pl.UInt16,
        },
    )
    df_expected = pl.LazyFrame(
        {
            "date": out_date,
            "ng": out_ng,
            "nf": out_nf,
            "sg": out_sg,
            "sf": out_sf,
            "tg": out_tg,
            "tf": out_tf,
        },
        schema={
            "date": pl.Date,
            "ng": pl.UInt8,
            "nf": pl.UInt16,
            "sg": pl.UInt8,
            "sf": pl.UInt16,
            "tg": pl.UInt8,
            "tf": pl.UInt16,
        },
    )
    df_out = seiryo_agg.calc_sn(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
        check_row_order=False,
    )


@pytest.mark.parametrize(
    "date_list",
    [
        [date(2020, 4, 5), date(2020, 4, 6), date(2020, 4, 8)],
        [date(2000, 4, 2), date(2001, 6, 3), date(2000, 2, 3)],
        [date(1969, 12, 25), date(1965, 8, 10), date(1970, 1, 1)],
    ],
)
def test_fill_sn(date_list: list[date]) -> None:
    df_in = pl.LazyFrame(
        {
            "date": date_list,
        },
        schema={
            "date": pl.Date,
        },
    )
    df_expected = pl.LazyFrame(
        {
            "date": date_list,
            "ng": [0] * len(date_list),
            "nf": [0] * len(date_list),
            "sg": [0] * len(date_list),
            "sf": [0] * len(date_list),
            "tg": [0] * len(date_list),
            "tf": [0] * len(date_list),
        },
        schema={
            "date": pl.Date,
            "ng": pl.UInt8,
            "nf": pl.UInt16,
            "sg": pl.UInt8,
            "sf": pl.UInt16,
            "tg": pl.UInt8,
            "tf": pl.UInt16,
        },
    )
    df_out = seiryo_agg.fill_sn(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
        check_row_order=False,
    )


def test_sort_ar_col_order() -> None:
    cols = ["date", "no", "lat_min", "lat_max", "lon_min", "lon_max"]
    df_in = pl.LazyFrame(
        {col_name: [] for col_name in sample(cols, len(cols))},
    )
    df_expected = pl.LazyFrame({col_name: [] for col_name in cols})
    df_out = seiryo_agg.sort_ar(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=True,
    )


@pytest.mark.parametrize(
    ("in_date", "in_no", "out_date", "out_no"),
    [
        (
            [date(2000, 4, 2), date(2001, 6, 3), date(2000, 2, 3)],
            [1, 1, 1],
            [date(2000, 2, 3), date(2000, 4, 2), date(2001, 6, 3)],
            [1, 1, 1],
        ),
        (
            [date(1969, 12, 25), date(1965, 8, 10), date(1970, 1, 1)],
            [1, 1, 1],
            [date(1965, 8, 10), date(1969, 12, 25), date(1970, 1, 1)],
            [1, 1, 1],
        ),
        (
            [date(2023, 4, 4), date(2023, 4, 4), date(2023, 4, 4)],
            [3, 1, 2],
            [date(2023, 4, 4), date(2023, 4, 4), date(2023, 4, 4)],
            [1, 2, 3],
        ),
    ],
)
def test_sort_ar_row_order(
    in_date: list[date],
    in_no: list[int],
    out_date: list[date],
    out_no: list[int],
) -> None:
    cols = ["lat_min", "lat_max", "lon_min", "lon_max"]
    df_in = pl.LazyFrame(
        {
            "date": in_date,
            "no": in_no,
        },
        schema={
            "date": pl.Date,
            "no": pl.UInt8,
        },
    ).with_columns([pl.lit(None).alias(col) for col in cols])
    df_expected = pl.LazyFrame(
        {
            "date": out_date,
            "no": out_no,
        },
        schema={
            "date": pl.Date,
            "no": pl.UInt8,
        },
    ).with_columns([pl.lit(None).alias(col) for col in cols])
    df_out = seiryo_agg.sort_ar(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
        check_row_order=True,
    )


def test_sort_sn_col_order() -> None:
    cols = ["date", "ng", "nf", "sg", "sf", "tg", "tf"]
    df_in = pl.LazyFrame(
        {col_name: [] for col_name in sample(cols, len(cols))},
    )
    df_expected = pl.LazyFrame({col_name: [] for col_name in cols})
    df_out = seiryo_agg.sort_sn(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=True,
    )


@pytest.mark.parametrize(
    ("in_date", "out_date"),
    [
        (
            [date(2000, 4, 2), date(2001, 6, 3), date(2000, 2, 3)],
            [date(2000, 2, 3), date(2000, 4, 2), date(2001, 6, 3)],
        ),
        (
            [date(1969, 12, 25), date(1965, 8, 10), date(1970, 1, 1)],
            [date(1965, 8, 10), date(1969, 12, 25), date(1970, 1, 1)],
        ),
    ],
)
def test_sort_sn_row_order(
    in_date: list[date],
    out_date: list[date],
) -> None:
    cols = ["ng", "nf", "sg", "sf", "tg", "tf"]
    df_in = pl.LazyFrame(
        {
            "date": in_date,
        },
        schema={
            "date": pl.Date,
        },
    ).with_columns([pl.lit(None).alias(col) for col in cols])
    df_expected = pl.LazyFrame(
        {
            "date": out_date,
        },
        schema={
            "date": pl.Date,
        },
    ).with_columns([pl.lit(None).alias(col) for col in cols])
    df_out = seiryo_agg.sort_sn(df_in)
    assert_frame_equal(
        df_out,
        df_expected,
        check_column_order=False,
        check_row_order=True,
    )
