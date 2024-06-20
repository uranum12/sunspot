from datetime import date
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pytest_mock import MockerFixture

import seiryo_sunspot_number_with_flare
import seiryo_sunspot_number_with_flare_config


@pytest.mark.parametrize(
    ("in_text", "out_date", "out_index"),
    [
        pytest.param(
            "Kandilli Observatory\n"
            "                              FLARE INDEX OF SOLAR ACTIVITY\n"
            "                                        FULL DISK\n"
            "      2003\n"
            "--------------------------------------------------------------------------------\n"
            "Day    Jan  Feb   Mar     Apr  May   Jun     Jul  Aug   Sep     Oct  Nov   Dec  \n"  # noqa: E501
            "================================================================================\n"
            "  1   1.11  1.48  0.55   2.31  8.81  0.74   0.68  0.00  0.00   3.33  7.21  0.00 \n"  # noqa: E501
            "  2   0.13  1.16  1.44   2.34  2.49  6.27   6.60  9.95  0.00   4.88 30.47  0.21 \n"  # noqa: E501
            "  3   5.17  0.00  0.00   3.52  0.46  0.58   3.70  3.28  0.17   1.12 15.79  0.00 \n"  # noqa: E501
            "  4   4.42  0.56  0.00  10.54  1.68  0.11   8.70  0.19  0.07   0.24 12.09  0.26 \n"  # noqa: E501
            "  5   2.44  0.57  0.73   3.03  0.92  0.74   1.81  0.80  0.00   0.90  0.41  0.11 \n"  # noqa: E501
            "================================================================================\n"
            "Mean  2.69  1.55  3.33   2.62  4.35  4.54   2.55  1.59  0.77  12.11  4.53  0.68 \n"  # noqa: E501
            "--------------------------------------------------------------------------------\n"
            "Yearly Mean = 3.46\n",
            [
                date(2003, 1, 1),
                date(2003, 2, 1),
                date(2003, 3, 1),
                date(2003, 4, 1),
                date(2003, 5, 1),
                date(2003, 6, 1),
                date(2003, 7, 1),
                date(2003, 8, 1),
                date(2003, 9, 1),
                date(2003, 10, 1),
                date(2003, 11, 1),
                date(2003, 12, 1),
            ],
            [
                2.69,
                1.55,
                3.33,
                2.62,
                4.35,
                4.54,
                2.55,
                1.59,
                0.77,
                12.11,
                4.53,
                0.68,
            ],
        ),
        pytest.param(
            "            Kandilli Observatory\n"
            "\n"
            "                            FLARE INDEX OF SOLAR ACTIVITY\n"
            "\n"
            "                                      FULL DISK\n"
            "\n"
            "2020\n"
            "\n"
            "-----------------------------------------------------------------------------------------------------\n"
            "Day\tJan\tFeb\tMar\tApr\tMay\tJun\tJul\tAug\tSep\tOct\tNov\tDec\n"
            "=====================================================================================================\n"
            ".....\n"
            "-----------------------------------------------------------------------------------------------------\n"
            "Mean\t0.01\t0.00\t0.00\t1.28\t0.00\t0.02\t0.00\t0.06\t0.00\t0.46\t1.08\t0.46\n"
            "-----------------------------------------------------------------------------------------------------\n"
            "Yearly Mean =\t0.28\n",
            [
                date(2020, 1, 1),
                date(2020, 2, 1),
                date(2020, 3, 1),
                date(2020, 4, 1),
                date(2020, 5, 1),
                date(2020, 6, 1),
                date(2020, 7, 1),
                date(2020, 8, 1),
                date(2020, 9, 1),
                date(2020, 10, 1),
                date(2020, 11, 1),
                date(2020, 12, 1),
            ],
            [
                0.01,
                0.00,
                0.00,
                1.28,
                0.00,
                0.02,
                0.00,
                0.06,
                0.00,
                0.46,
                1.08,
                0.46,
            ],
        ),
    ],
)
def test_load_flare_file(
    mocker: MockerFixture,
    in_text: str,
    out_date: list[date],
    out_index: list[float],
) -> None:
    m = mocker.mock_open(read_data=in_text)
    mocker.patch("pathlib.Path.open", m)

    df_expected = pl.DataFrame(
        {"date": out_date, "index": out_index},
        schema={"date": pl.Date, "index": pl.Float64},
    )

    df_out = seiryo_sunspot_number_with_flare.load_flare_file(
        Path("dummy/path")
    )

    assert_frame_equal(df_out, df_expected, check_column_order=False)


def test_load_flare_data(mocker: MockerFixture) -> None:
    mocker.patch(
        "pathlib.Path.glob",
        side_effect=[
            (Path(f"north_{year}") for year in range(2010, 2012)),
            (Path(f"south_{year}") for year in range(2010, 2012)),
            (Path(f"total_{year}") for year in range(2011, 2013)),
        ],
    )

    def _create_df(dt: date, v: float) -> pl.DataFrame:
        return pl.DataFrame(
            {"date": [dt], "index": [v]},
            schema={"date": pl.Date, "index": pl.Float64},
        )

    mocker.patch(
        "seiryo_sunspot_number_with_flare.load_flare_file",
        side_effect=[
            _create_df(date(2010, 1, 1), 1.2),
            _create_df(date(2011, 1, 1), 2.3),
            _create_df(date(2010, 1, 1), 12.3),
            _create_df(date(2011, 1, 1), 23.4),
            _create_df(date(2011, 1, 1), 12),
            _create_df(date(2012, 1, 1), 23),
        ],
    )
    df_expected = pl.DataFrame(
        {
            "date": [date(2010, 1, 1), date(2011, 1, 1), date(2012, 1, 1)],
            "north": [1.2, 2.3, None],
            "south": [12.3, 23.4, None],
            "total": [None, 12, 23],
        },
        schema={
            "date": pl.Date,
            "north": pl.Float64,
            "south": pl.Float64,
            "total": pl.Float64,
        },
    )
    df_out = seiryo_sunspot_number_with_flare.load_flare_data(
        Path("bummy/path")
    )
    print(df_out)
    assert_frame_equal(df_out, df_expected, check_column_order=False)


@pytest.mark.parametrize(
    (
        "in_seiryo_date",
        "in_seiryo_north",
        "in_seiryo_south",
        "in_seiryo_total",
        "in_flare_date",
        "in_flare_north",
        "in_flare_south",
        "in_flare_total",
        "out_date",
        "out_seiryo_north",
        "out_seiryo_south",
        "out_seiryo_total",
        "out_flare_north",
        "out_flare_south",
        "out_flare_total",
    ),
    [
        pytest.param(
            [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
            [2, 4, 6],
            [3, 5, 7],
            [4, 6, 8],
            [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [2, 4, 6],
            [3, 5, 7],
            [4, 6, 8],
        ),
        pytest.param(
            [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [
                date(2020, 1, 1),
                date(2020, 2, 1),
                date(2020, 3, 1),
                date(2020, 4, 1),
                date(2020, 5, 1),
            ],
            [2, 4, 6, 8, 10],
            [3, 5, 7, 9, 11],
            [4, 6, 8, 10, 12],
            [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [2, 4, 6],
            [3, 5, 7],
            [4, 6, 8],
        ),
        pytest.param(
            [
                date(2020, 1, 1),
                date(2020, 2, 1),
                date(2020, 3, 1),
                date(2020, 4, 1),
                date(2020, 5, 1),
            ],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
            [date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1)],
            [2, 4, 6],
            [3, 5, 7],
            [4, 6, 8],
            [
                date(2020, 1, 1),
                date(2020, 2, 1),
                date(2020, 3, 1),
                date(2020, 4, 1),
                date(2020, 5, 1),
            ],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
            [None, 2, 4, 6, None],
            [None, 3, 5, 7, None],
            [None, 4, 6, 8, None],
        ),
    ],
)
def test_join_data(
    in_seiryo_date: list[date],
    in_seiryo_north: list[float],
    in_seiryo_south: list[float],
    in_seiryo_total: list[float],
    in_flare_date: list[date],
    in_flare_north: list[float],
    in_flare_south: list[float],
    in_flare_total: list[float],
    out_date: list[date],
    out_seiryo_north: list[float],
    out_seiryo_south: list[float],
    out_seiryo_total: list[float],
    out_flare_north: list[float],
    out_flare_south: list[float],
    out_flare_total: list[float],
) -> None:
    df_in_seiryo = pl.DataFrame(
        {
            "date": in_seiryo_date,
            "north": in_seiryo_north,
            "south": in_seiryo_south,
            "total": in_seiryo_total,
        },
        schema={
            "date": pl.Date,
            "north": pl.Float64,
            "south": pl.Float64,
            "total": pl.Float64,
        },
    )
    df_in_flare = pl.DataFrame(
        {
            "date": in_flare_date,
            "north": in_flare_north,
            "south": in_flare_south,
            "total": in_flare_total,
        },
        schema={
            "date": pl.Date,
            "north": pl.Float64,
            "south": pl.Float64,
            "total": pl.Float64,
        },
    )
    df_expected = pl.DataFrame(
        {
            "date": out_date,
            "seiryo_north": out_seiryo_north,
            "seiryo_south": out_seiryo_south,
            "seiryo_total": out_seiryo_total,
            "flare_north": out_flare_north,
            "flare_south": out_flare_south,
            "flare_total": out_flare_total,
        },
        schema={
            "date": pl.Date,
            "seiryo_north": pl.Float64,
            "seiryo_south": pl.Float64,
            "seiryo_total": pl.Float64,
            "flare_north": pl.Float64,
            "flare_south": pl.Float64,
            "flare_total": pl.Float64,
        },
    )
    df_out = seiryo_sunspot_number_with_flare.join_data(
        df_in_seiryo, df_in_flare
    )
    assert_frame_equal(df_out, df_expected)


def test_calc_factors() -> None:
    in_df = pl.DataFrame(
        {
            "seiryo_north": [1.2, 2.2, 2.1],
            "seiryo_south": [2.3, 3.3, 3.2],
            "seiryo_total": [3.4, 4.4, 4.3],
            "flare_north": [4.5, None, 5.4],
            "flare_south": [5.6, 6.6, 6.5],
            "flare_total": [6.7, 7.7, 7.6],
        },
        schema={
            "seiryo_north": pl.Float64,
            "seiryo_south": pl.Float64,
            "seiryo_total": pl.Float64,
            "flare_north": pl.Float64,
            "flare_south": pl.Float64,
            "flare_total": pl.Float64,
        },
    )
    _ = seiryo_sunspot_number_with_flare.calc_factors(in_df)


def test_draw_sunspot_number_with_flare() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1)],
            "seiryo_total": [1, 2, 3],
            "flare_total": [1, 2, 3],
        },
        schema={
            "date": pl.Date,
            "seiryo_total": pl.Float64,
            "flare_total": pl.Float64,
        },
    )
    config = seiryo_sunspot_number_with_flare_config.SunspotNumberWithFlare()
    _ = seiryo_sunspot_number_with_flare.draw_sunspot_number_with_flare(
        df, config
    )
    _ = seiryo_sunspot_number_with_flare.draw_sunspot_number_with_flare(
        df, config, factor=1.0
    )


def test_draw_sunspot_number_with_flare_hemispheric() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1)],
            "seiryo_north": [1, 2, 3],
            "seiryo_south": [1, 2, 3],
            "flare_north": [1, 2, 3],
            "flare_south": [1, 2, 3],
        },
        schema={
            "date": pl.Date,
            "seiryo_north": pl.Float64,
            "seiryo_south": pl.Float64,
            "flare_north": pl.Float64,
            "flare_south": pl.Float64,
        },
    )
    config = seiryo_sunspot_number_with_flare_config.SunspotNumberWithFlareHemispheric()  # noqa: E501
    _ = seiryo_sunspot_number_with_flare.draw_sunspot_number_with_flare_hemispheric(  # noqa: E501
        df, config
    )
    _ = seiryo_sunspot_number_with_flare.draw_sunspot_number_with_flare_hemispheric(  # noqa: E501
        df, config, factor_north=1.0, factor_south=1.0
    )
