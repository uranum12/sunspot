from io import BytesIO

import polars as pl
import streamlit as st
from matplotlib.figure import Figure


def download_df(df: pl.DataFrame, file_stem: str) -> None:
    file_parquet = BytesIO()
    df.write_parquet(file_parquet)
    file_parquet.seek(0)
    with st.container():
        st.caption("ダウンロード")
        col_button, col_option = st.columns(2)
        with col_option:
            file_type = st.selectbox(
                "ダウンロード形式",
                ("CSV", "Parquet"),
                key=file_stem,
            )
        with col_button:
            match file_type:
                case "CSV":
                    st.download_button(
                        "ダウンロード",
                        df.write_csv(),
                        file_name=f"{file_stem}.csv",
                        help="データをCSV形式でダウンロードします",
                    )
                case "Parquet":
                    st.download_button(
                        "ダウンロード",
                        file_parquet,
                        file_name=f"{file_stem}.parquet",
                        help="データをParquet形式でダウンロードします",
                    )


def download_figure(fig: Figure, file_stem: str) -> None:
    with st.container():
        st.caption("ダウンロード")
        col_button, col_option = st.columns(2)
        with col_option:
            file_type = st.selectbox(
                "ダウンロード形式",
                ("png", "jpg", "pdf"),
                key=file_stem + "_download_type",
            )
            dpi = st.number_input(
                "dpi",
                min_value=100,
                max_value=1000,
                value=300,
                step=100,
                key=file_stem + "_number",
            )
        with col_button:
            file = BytesIO()
            with st.spinner("wait....."):
                fig.savefig(
                    file,
                    format=file_type,
                    dpi=dpi,
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
            file.seek(0)
            st.download_button(
                "ダウンロード",
                file,
                f"{file_stem}.{file_type}",
            )
