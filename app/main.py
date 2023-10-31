import polars as pl
import streamlit as st

from components.download_button import download_df, download_figure
from components.show_detail import show_detail_all, show_detail_wolf
from libs import agg, wolf


def main() -> None:
    uploaded_files = st.file_uploader(
        "ファイルを選択",
        type="csv",
        accept_multiple_files=True,
        help="CSV形式でデータが記載されたファイルをアップロードします",
    )
    if uploaded_files is not None:
        dfl: list[pl.DataFrame] = []
        for uploaded_file in uploaded_files:
            df = pl.read_csv(uploaded_file, infer_schema_length=0)
            df = agg.fill_date(df)
            dfl.append(df)
        if len(dfl) > 0:
            with st.spinner("計算中....."):
                df = (
                    pl.concat(dfl)
                    .lazy()
                    .pipe(agg.convert_number)
                    .pipe(agg.convert_date)
                    .pipe(agg.convert_lat)
                    .pipe(agg.sort)
                    .collect()
                )
                df_wolf = df.pipe(wolf.calc_lat)
                df_wolf_daily = df_wolf.lazy().pipe(wolf.daily).collect()
                df_wolf_monthly = (
                    df_wolf_daily.lazy().pipe(wolf.monthly).collect()
                )

            st.header("データ")
            tab_data_all, tab_data_wolf, tab_data_butter = st.tabs(
                ["全体", "黒点相対数", "蝶形図"],
            )

            with tab_data_all:
                st.subheader("全体")
                show_detail_all(df)
                download_df(df, "all")

            with tab_data_wolf:
                st.subheader("1日ごと")
                show_detail_wolf(df_wolf_daily, decimal=False)
                download_df(df_wolf_daily, "wolf_daily")
                st.subheader("1月ごと")
                show_detail_wolf(df_wolf_monthly, day=False)
                download_df(df_wolf_monthly, "wolf_monthly")

            with tab_data_butter:
                st.warning("準備中...")

            st.header("グラフ")
            tab_graph_wolf, tab_graph_butter = st.tabs(["黒点相対数", "蝶形図"])

            with st.spinner("描画中....."):
                graph_wolf_daily = wolf.graph_daily(df_wolf_daily)
                graph_wolf_monthly = wolf.graph_monthly(df_wolf_monthly)

            with tab_graph_wolf:
                tab_daily, tab_monthly = st.tabs(["1日ごと", "1月ごと"])
                with tab_daily:
                    st.pyplot(graph_wolf_daily)
                    download_figure(graph_wolf_daily, "wolf_daily")
                with tab_monthly:
                    st.pyplot(graph_wolf_monthly)
                    download_figure(graph_wolf_monthly, "wolf_monthly")

            with tab_graph_butter:
                st.warning("準備中...")


if __name__ == "__main__":
    main()
