from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import polars as pl
from dateutil.relativedelta import relativedelta
from matplotlib.figure import Figure


def calc_date_range(df: pl.LazyFrame) -> tuple[date, date]:
    """日付の開始日と最終日を算出する

    Args:
        df (pl.LazyFrame): 黒点数データ

    Returns:
        tuple[date, date]: 開始日と最終日
    """
    date_range: dict[str, date] = (
        df.select(
            pl.min("date").alias("start"),
            pl.max("date").alias("end"),
        )
        .collect()
        .to_dicts()[0]
    )
    return date_range["start"], date_range["end"]


def adjust_dates(start: date, end: date) -> tuple[date, date]:
    """日付の範囲を月ごとに調整する

    Args:
        start (date): 開始日
        end (date): 最終日

    Returns:
        tuple[date, date]: 開始日と最終日
    """
    start = start.replace(day=1)
    end += relativedelta(months=1, day=1, days=-1)
    return start, end


def calc_dayly_obs(df: pl.LazyFrame, start: date, end: date) -> pl.LazyFrame:
    """日ごとの観測日数を算出する

    Args:
        df (pl.LazyFrame): 黒点数データ
        start (date): 開始日
        end (date): 終了日

    Returns:
        pl.LazyFrame: 日ごとの観測
    """

    return (
        pl.LazyFrame(
            {"date": pl.date_range(start, end, interval="1d", eager=True)},
        )
        .join(
            df.select("date").with_columns(pl.lit(1).alias("obs")),
            on="date",
            how="left",
        )
        .with_columns(pl.col("obs").fill_null(0).cast(pl.UInt8))
    )


def calc_monthly_obs(df: pl.LazyFrame) -> pl.LazyFrame:
    """月ごとの観測日数を算出する

    Args:
        df (pl.LazyFrame): 日ごとの観測日数データ

    Returns:
        pl.LazyFrame: 月ごとの観測日数
    """
    return (
        df.with_columns(pl.col("date").dt.truncate("1mo"))
        .group_by("date")
        .agg(pl.col("obs").sum().cast(pl.UInt8))
        .sort("date")
    )


def draw_monthly_obs_days(df: pl.DataFrame) -> Figure:
    """月ごとの観測日数のグラフを作成する

    Args:
        df (pl.DataFrame): 月ごとの観測日数

    Returns:
        Figure: 作成したグラフ
    """
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.bar(df["date"], df["obs"], width=15)

    ax.set_title("observations days per month")
    ax.set_xlabel("date")
    ax.set_ylabel("observations days")

    ax.xaxis.set_major_locator(locator := mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()

    return fig


def draw_monthly_obs_days_plotly(df: pl.DataFrame) -> go.Figure:
    return (
        go.Figure()
        .add_trace(
            go.Bar(
                x=df["date"],
                y=df["obs"],
            ),
        )
        .update_layout(
            {
                "title": {
                    "text": "observations days per month",
                },
                "xaxis": {
                    "title": {
                        "text": "date",
                    },
                },
                "yaxis": {
                    "title": {
                        "text": "observations days",
                    },
                },
            },
        )
    )


def main() -> None:
    data_file = Path("out/seiryo/sn.parquet")
    output_path = Path("out/seiryo")

    df = pl.scan_parquet(data_file)

    start, end = adjust_dates(*calc_date_range(df))

    df_daily = calc_dayly_obs(df, start, end).collect()
    print(df_daily)
    df_daily.write_parquet(output_path / "observations_daily.parquet")

    df_monthly = calc_monthly_obs(df_daily.lazy()).collect()
    print(df_monthly)
    df_monthly.write_parquet(output_path / "observations_monthly.parquet")

    fig = draw_monthly_obs_days_plotly(df_monthly)
    fig.update_layout(
        {
            "template": "simple_white",
            "font_family": "Century",
            "title": {
                "font_size": 24,
                "x": 0.5,
                "y": 0.9,
                "xanchor": "center",
                "yanchor": "middle",
            },
            "xaxis": {
                "title_font_size": 20,
                "tickfont_size": 16,
                "linewidth": 1,
                "mirror": True,
                "showgrid": True,
                "ticks": "outside",
            },
            "yaxis": {
                "title_font_size": 20,
                "tickfont_size": 16,
                "linewidth": 1,
                "mirror": True,
                "showgrid": True,
                "ticks": "outside",
            },
        },
    )
    fig.write_json(
        output_path / "observations_days.json",
        pretty=True,
    )
    for ext in "pdf", "png":
        file_path = output_path / f"observations_days.{ext}"
        fig.write_image(
            file_path,
            width=800,
            height=500,
            engine="kaleido",
            scale=10,
        )


if __name__ == "__main__":
    main()
