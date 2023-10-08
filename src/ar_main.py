from collections import defaultdict
from pathlib import Path
from pprint import pprint

import polars as pl

import ar_common
import ar_merge
import ar_new
import ar_notebook
import ar_old
import ar_type


def calc_obs_date(df: pl.LazyFrame, year: int, month: int) -> pl.LazyFrame:
    match ar_type.detect_schema_type(year, month):
        case (
            ar_type.SchemaType.NOTEBOOK_1
            | ar_type.SchemaType.NOTEBOOK_2
            | ar_type.SchemaType.NOTEBOOK_3
        ):
            df = ar_notebook.calc_obs_date(df, year, month)
            df = ar_notebook.fill_blanks(df, [("last", pl.Date)])
        case ar_type.SchemaType.OLD:
            df = ar_old.calc_obs_date(df, year, month)
        case ar_type.SchemaType.NEW:
            df = ar_new.calc_obs_date(df, year, month)
    return df


def calc_no(
    df: pl.LazyFrame,
    schema_type: ar_type.SchemaType,
) -> pl.LazyFrame:
    match schema_type:
        case ar_type.SchemaType.NOTEBOOK_1:
            df = ar_common.convert_no(df)
        case ar_type.SchemaType.NOTEBOOK_3:
            df = ar_common.extract_ns(df)
            df = ar_notebook.concat_no(df)
            df = ar_common.convert_no(df)
        case (
            ar_type.SchemaType.NOTEBOOK_2
            | ar_type.SchemaType.OLD
            | ar_type.SchemaType.NEW
        ):
            df = ar_common.extract_ns(df)
            df = ar_common.convert_no(df)
    return df


def calc_coords(
    df: pl.LazyFrame,
    schema_type: ar_type.SchemaType,
) -> pl.LazyFrame:
    match schema_type:
        case ar_type.SchemaType.NOTEBOOK_1 | ar_type.SchemaType.NOTEBOOK_2:
            df = ar_common.extract_coords_lr(df, ["lat"])
            df = ar_common.convert_lat(df)
            df = ar_notebook.fill_blanks(
                df,
                [
                    ("over", pl.Boolean),
                    ("lon_left", pl.UInt16),
                    ("lon_right", pl.UInt16),
                    ("lat_left_sign", pl.Categorical),
                    ("lat_right_sign", pl.Categorical),
                    ("lon_left_sign", pl.Categorical),
                    ("lon_right_sign", pl.Categorical),
                    ("lat_question", pl.Categorical),
                    ("lon_question", pl.Categorical),
                ],
            )
        case ar_type.SchemaType.NOTEBOOK_3:
            df = ar_common.extract_coords_lr(df, ["lat"])
            df = ar_common.extract_coords_sign(df, ["lat"])
            df = ar_common.convert_lat(df)
            df = ar_notebook.fill_blanks(
                df,
                [
                    ("over", pl.Boolean),
                    ("lon_left", pl.UInt16),
                    ("lon_right", pl.UInt16),
                    ("lon_left_sign", pl.Categorical),
                    ("lon_right_sign", pl.Categorical),
                    ("lat_question", pl.Categorical),
                    ("lon_question", pl.Categorical),
                ],
            )
        case ar_type.SchemaType.OLD | ar_type.SchemaType.NEW:
            df = ar_common.detect_coords_over(df)
            df = ar_common.extract_coords_qm(df)
            df = ar_common.extract_coords_lr(df)
            df = ar_common.extract_coords_sign(df)
            df = ar_common.convert_lat(df)
            df = ar_common.convert_lon(df)
    return df


def main() -> None:
    # 入出力先のフォルダのパス
    # 出力先のフォルダがない場合は作成
    data_path = Path("data/fujimori_ar")
    output_path = Path("out/ar")
    output_path.mkdir(parents=True, exist_ok=True)

    # 形式ごとに日付の計算
    dfl_by_schema = defaultdict(list)
    for path in data_path.glob("*-*.csv"):
        year, month = map(int, path.stem.split("-"))
        if (schema_type := ar_type.detect_schema_type(year, month)) is None:
            print(f"Err: not supported date for {year}/{month}")
            continue
        df = pl.scan_csv(path, dtypes=ar_type.detect_dtypes(schema_type))
        df = calc_obs_date(df, year, month)
        dfl_by_schema[schema_type].append(df)

    # 形式ごとに一つのデータフレームへ結合
    df_by_schema = {
        schema_type: pl.concat(dfl)
        for schema_type, dfl in dfl_by_schema.items()
    }

    # 形式ごとに通し番号の計算
    df_by_schema = {
        schema_type: calc_no(df, schema_type)
        for schema_type, df in df_by_schema.items()
    }

    # 形式ごとに経緯度の計算
    df_by_schema = {
        schema_type: calc_coords(df, schema_type)
        for schema_type, df in df_by_schema.items()
    }

    # 複数のシートに跨って存在するデータを一つに結合
    df_merged = ar_merge.merge(
        pl.concat(
            [
                df_by_schema[ar_type.SchemaType.OLD],
                df_by_schema[ar_type.SchemaType.NEW],
            ],
        ),
    )

    # 全ての処理済みを一つのデータフレームへ結合しソート
    df_sorted = ar_common.sort(
        pl.concat(
            [
                df_by_schema[ar_type.SchemaType.NOTEBOOK_1],
                df_by_schema[ar_type.SchemaType.NOTEBOOK_2],
                df_by_schema[ar_type.SchemaType.NOTEBOOK_3],
                df_merged,
            ],
        ),
    )

    with pl.StringCache():
        # 中間結果を計算し保存
        for schema_type, df in df_by_schema.items():
            file_name = output_path / f"{schema_type.name.lower()}.parquet"
            df.collect().write_parquet(file_name)
        df_merged.collect().write_parquet(output_path / "merged.parquet")
        # プロファイルとともに計算
        df_all, profile = df_sorted.profile()

    # 結果を表示し、保存
    pprint(df_all.schema)
    print(df_all)
    print(profile)
    df_all.write_parquet(output_path / "all.parquet")
    profile.write_csv(output_path / "profile.csv")


if __name__ == "__main__":
    main()
