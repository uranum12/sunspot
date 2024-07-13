from datetime import date

import numpy as np
import polars as pl
import pytest

import seiryo_butterfly
import seiryo_butterfly_merge


def test_color() -> None:
    color = seiryo_butterfly_merge.Color(2, 4, 6)
    dict_expected = {"red": 2, "green": 4, "blue": 6}
    tuple_expected = (2, 4, 6)
    assert color.to_dict() == dict_expected
    assert color.to_tuple() == tuple_expected


@pytest.mark.parametrize(
    ("in_red", "in_green", "in_blue"),
    [
        pytest.param(1000, 10, 10),
        pytest.param(-10, 10, 10),
        pytest.param(10, 1000, 10),
        pytest.param(10, -10, 10),
        pytest.param(10, 10, 1000),
        pytest.param(10, 10, -10),
    ],
)
def test_color_with_error(in_red: int, in_green: int, in_blue: int) -> None:
    with pytest.raises(ValueError, match="color value must be 0x00 to 0xFF"):
        _ = seiryo_butterfly_merge.Color(in_red, in_green, in_blue)


def test_merge_info() -> None:
    info1 = seiryo_butterfly.ButterflyInfo(
        -10,
        50,
        date(2020, 1, 1),
        date(2020, 2, 2),
        seiryo_butterfly.DateDelta(months=1),
    )
    info2 = seiryo_butterfly.ButterflyInfo(
        -40,
        40,
        date(2010, 5, 1),
        date(2011, 12, 1),
        seiryo_butterfly.DateDelta(months=1),
    )
    info_expected = seiryo_butterfly.ButterflyInfo(
        -40,
        50,
        date(2010, 5, 1),
        date(2020, 2, 2),
        seiryo_butterfly.DateDelta(months=1),
    )
    info_merged = seiryo_butterfly_merge.merge_info([info1, info2])
    assert info_merged == info_expected


def test_merge_info_with_error() -> None:
    info1 = seiryo_butterfly.ButterflyInfo(
        -10,
        50,
        date(2020, 1, 1),
        date(2020, 2, 2),
        seiryo_butterfly.DateDelta(months=1),
    )
    info2 = seiryo_butterfly.ButterflyInfo(
        -40,
        40,
        date(2010, 5, 1),
        date(2011, 12, 1),
        seiryo_butterfly.DateDelta(days=1),
    )
    with pytest.raises(ValueError, match="Date interval must be equal"):
        _ = seiryo_butterfly_merge.merge_info([info1, info2])


@pytest.mark.parametrize(
    ("in_data", "out_img"),
    [
        pytest.param(
            [
                {
                    "date": [
                        date(2020, 2, 1),
                        date(2020, 2, 2),
                        date(2020, 2, 3),
                        date(2020, 2, 4),
                        date(2020, 2, 5),
                    ],
                    "min": [[], [-2, 1], [0], [-1, 1], [-1]],
                    "max": [[], [-1, 2], [1], [-1, 1], [2]],
                },
                {
                    "date": [
                        date(2020, 2, 1),
                        date(2020, 2, 2),
                        date(2020, 2, 3),
                        date(2020, 2, 4),
                        date(2020, 2, 5),
                    ],
                    "min": [[0], [0], [-4], [-1, 1], [0]],
                    "max": [[0], [1], [-2], [0, 2], [1]],
                },
            ],
            [
                [0, 1, 0, 2, 1],  # +2
                [0, 1, 0, 2, 1],
                [0, 3, 1, 3, 3],  # +1
                [0, 2, 1, 0, 3],
                [2, 2, 1, 2, 3],  # 0
                [0, 0, 0, 2, 1],
                [0, 1, 0, 3, 1],  # -1
                [0, 1, 0, 0, 0],
                [0, 1, 2, 0, 0],  # -2
            ],
        ),
        pytest.param(
            [
                {
                    "date": [date(2020, 2, 1), date(2020, 2, 2)],
                    "min": [[1], [1]],
                    "max": [[1], [1]],
                },
                {
                    "date": [date(2020, 2, 2), date(2020, 2, 3)],
                    "min": [[0], [0]],
                    "max": [[0], [0]],
                },
                {
                    "date": [date(2020, 2, 3), date(2020, 2, 4)],
                    "min": [[-1], [-1]],
                    "max": [[-1], [-1]],
                },
                {
                    "date": [date(2020, 2, 4), date(2020, 2, 5)],
                    "min": [[-2], [-2]],
                    "max": [[-2], [-2]],
                },
                {
                    "date": [date(2020, 2, 5), date(2020, 2, 6)],
                    "min": [[2], [2]],
                    "max": [[2], [2]],
                },
            ],
            [
                [0, 0, 0, 0, 16],  # +2
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],  # +1
                [0, 0, 0, 0, 0],
                [0, 2, 2, 0, 0],  # 0
                [0, 0, 0, 0, 0],
                [0, 0, 4, 4, 0],  # -1
                [0, 0, 0, 0, 0],
                [0, 0, 0, 8, 8],  # -2
            ],
        ),
    ],
)
def test_create_merged_image(
    in_data: list[dict[str, list]], out_img: list[list[int]]
) -> None:
    dfl_in = [
        pl.DataFrame(
            data,
            schema={
                "date": pl.Date,
                "min": pl.List(pl.Int8),
                "max": pl.List(pl.Int8),
            },
        )
        for data in in_data
    ]
    info = seiryo_butterfly.ButterflyInfo.from_dict(
        {
            "lat_min": -2,
            "lat_max": 2,
            "date_start": "2020-02-01",
            "date_end": "2020-02-05",
            "date_interval": "P1D",
        }
    )
    out = seiryo_butterfly_merge.create_merged_image(dfl_in, info)
    np.testing.assert_equal(out, out_img)


@pytest.mark.parametrize(
    ("in_img", "in_cmap", "out_img"),
    [
        pytest.param(
            [[0, 0, 0], [0, 1, 2], [3, 2, 1]],
            [
                seiryo_butterfly_merge.Color(0xFF, 0x00, 0x00),
                seiryo_butterfly_merge.Color(0x00, 0xFF, 0x00),
                seiryo_butterfly_merge.Color(0x00, 0x00, 0xFF),
            ],
            [
                [[0xFF, 0xFF, 0xFF], [0xFF, 0xFF, 0xFF], [0xFF, 0xFF, 0xFF]],
                [[0xFF, 0xFF, 0xFF], [0xFF, 0x00, 0x00], [0x00, 0xFF, 0x00]],
                [[0x00, 0x00, 0xFF], [0x00, 0xFF, 0x00], [0xFF, 0x00, 0x00]],
            ],
        ),
        pytest.param(
            [[1, 2, 4], [1, 2, 4], [1, 2, 4]],
            [
                seiryo_butterfly_merge.Color(0xFF, 0x00, 0x00),
                seiryo_butterfly_merge.Color(0x00, 0xFF, 0x00),
                seiryo_butterfly_merge.Color(0x00, 0x00, 0xFF),
                seiryo_butterfly_merge.Color(0xFF, 0xFF, 0x00),
                seiryo_butterfly_merge.Color(0xFF, 0x00, 0xFF),
                seiryo_butterfly_merge.Color(0x00, 0xFF, 0xFF),
            ],
            [
                [[0xFF, 0x00, 0x00], [0x00, 0xFF, 0x00], [0xFF, 0xFF, 0x00]],
                [[0xFF, 0x00, 0x00], [0x00, 0xFF, 0x00], [0xFF, 0xFF, 0x00]],
                [[0xFF, 0x00, 0x00], [0x00, 0xFF, 0x00], [0xFF, 0xFF, 0x00]],
            ],
        ),
    ],
)
def test_create_color_image(
    in_img: list[list[int]],
    in_cmap: list[seiryo_butterfly_merge.Color],
    out_img: list[list[list[int]]],
) -> None:
    out = seiryo_butterfly_merge.create_color_image(
        np.array(in_img, dtype=np.uint16), in_cmap
    )
    np.testing.assert_equal(out, out_img)
