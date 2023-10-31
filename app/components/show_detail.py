import polars as pl
import streamlit as st


def show_detail_all(df: pl.DataFrame) -> None:
    with st.expander("詳細"):
        st.dataframe(
            df,
            column_config={
                "date": st.column_config.DateColumn(
                    "date",
                    format="Y/M/D",
                    help="観測日",
                ),
                "no": st.column_config.NumberColumn(
                    "no",
                    help="黒点群の経緯度を測る時に割り振った番号",
                ),
                "num": st.column_config.NumberColumn(
                    "num",
                    help="黒点群に含まれている黒点の数",
                ),
                "lat_max": st.column_config.NumberColumn(
                    "lat_max",
                    help="黒点群の経緯度の北側",
                ),
                "lat_min": st.column_config.NumberColumn(
                    "lat_min",
                    help="黒点群の経緯度の南側",
                ),
                "lon_max": st.column_config.NumberColumn(
                    "lon_max",
                    help="黒点群の経緯度の東側",
                ),
                "lon_min": st.column_config.NumberColumn(
                    "lon_min",
                    help="黒点群の経緯度の西側",
                ),
            },
            use_container_width=True,
        )


def show_detail_wolf(
    df: pl.DataFrame,
    *,
    day: bool = True,
    decimal: bool = True,
) -> None:
    date_format = "Y/M/D" if day else "Y/M"
    num_format = "%.2f" if decimal else "%d"
    with st.expander("詳細"):
        st.dataframe(
            df,
            column_config={
                "date": st.column_config.DateColumn(
                    "date",
                    format=date_format,
                    help="観測日",
                ),
                "ng": st.column_config.NumberColumn(
                    "ng",
                    format=num_format,
                    help="北半球の黒点群の個数",
                ),
                "nf": st.column_config.NumberColumn(
                    "nf",
                    format=num_format,
                    help="北半球の黒点の数の合計",
                ),
                "nr": st.column_config.NumberColumn(
                    "nr",
                    format=num_format,
                    help="北半球の $R=10g+f$ で求められる黒点相対数",
                ),
                "sg": st.column_config.NumberColumn(
                    "sg",
                    format=num_format,
                    help="南半球の黒点群の個数",
                ),
                "sf": st.column_config.NumberColumn(
                    "sf",
                    format=num_format,
                    help="南半球の黒点の数の合計",
                ),
                "sr": st.column_config.NumberColumn(
                    "sr",
                    format=num_format,
                    help="南半球の $R=10g+f$ で求められる黒点相対数",
                ),
                "tg": st.column_config.NumberColumn(
                    "tg",
                    format=num_format,
                    help="黒点群の個数",
                ),
                "tf": st.column_config.NumberColumn(
                    "tf",
                    format=num_format,
                    help="黒点の数の合計",
                ),
                "tr": st.column_config.NumberColumn(
                    "tr",
                    format=num_format,
                    help="$R=10g+f$ で求められる黒点相対数",
                ),
            },
            use_container_width=True,
        )
