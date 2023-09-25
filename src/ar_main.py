from pathlib import Path
from pprint import pprint

import polars as pl

import ar_common
import ar_merge
import ar_new
import ar_notebook
import ar_old
import ar_type


def main() -> None:  # noqa: C901, PLR0915
    # 入出力先のフォルダのパス
    # 出力先のフォルダがない場合は作成
    data_path = Path("data/fujimori_ar")
    output_path = Path("out/ar")
    output_path.mkdir(parents=True, exist_ok=True)

    # 形式ごとに日付の計算
    dfl_by_schema: dict[ar_type.SchemaType, list[pl.LazyFrame]] = {
        schema_type: [] for schema_type in ar_type.SchemaType
    }
    for path in data_path.glob("*-*.csv"):
        year, month = map(int, path.stem.split("-"))
        if (schema_type := ar_type.detect_schema_type(year, month)) is None:
            print(f"Err: not supported date for {year}/{month}")
            continue
        df = pl.scan_csv(path, dtypes=ar_type.detect_dtypes(schema_type))
        match schema_type:
            case (
                ar_type.SchemaType.NOTEBOOK_1
                | ar_type.SchemaType.NOTEBOOK_2
                | ar_type.SchemaType.NOTEBOOK_3
            ):
                df = ar_notebook.calc_obs_date(df, year, month)
            case ar_type.SchemaType.OLD:
                df = ar_old.calc_obs_date(df, year, month)
            case ar_type.SchemaType.NEW:
                df = ar_new.calc_obs_date(df, year, month)
        dfl_by_schema[schema_type].append(df)

    # 形式ごとに一つのデータフレームへ結合
    df_by_schema: dict[ar_type.SchemaType, pl.LazyFrame] = {}
    for schema_type, dfl in dfl_by_schema.items():
        df_by_schema[schema_type] = pl.concat(dfl)

    # 形式ごとに通し番号の計算
    for schema_type in df_by_schema:
        df = df_by_schema[schema_type]
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
        df_by_schema[schema_type] = df

    # 形式ごとに経緯度の計算
    for schema_type in df_by_schema:
        df = df_by_schema[schema_type]
        match schema_type:
            case ar_type.SchemaType.NOTEBOOK_1 | ar_type.SchemaType.NOTEBOOK_2:
                df = ar_common.extract_coords_lr(df, ["lat"])
                df = ar_common.convert_lat(df)
            case ar_type.SchemaType.NOTEBOOK_3:
                df = ar_common.extract_coords_lr(df, ["lat"])
                df = ar_common.extract_coords_sign(df, ["lat"])
                df = ar_common.convert_lat(df)
            case ar_type.SchemaType.OLD | ar_type.SchemaType.NEW:
                df = ar_common.detect_coords_over(df)
                df = ar_common.extract_coords_qm(df)
                df = ar_common.extract_coords_lr(df)
                df = ar_common.extract_coords_sign(df)
                df = ar_common.convert_lat(df)
                df = ar_common.convert_lon(df)
        df_by_schema[schema_type] = df

    for schema_type in df_by_schema:
        df = df_by_schema[schema_type]
        match schema_type:
            case ar_type.SchemaType.NOTEBOOK_1 | ar_type.SchemaType.NOTEBOOK_2:
                df = ar_notebook.fill_blanks(
                    df,
                    [
                        ("last", pl.Date),
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
                df = ar_notebook.fill_blanks(
                    df,
                    [
                        ("last", pl.Date),
                        ("over", pl.Boolean),
                        ("lon_left", pl.UInt16),
                        ("lon_right", pl.UInt16),
                        ("lon_left_sign", pl.Categorical),
                        ("lon_right_sign", pl.Categorical),
                        ("lat_question", pl.Categorical),
                        ("lon_question", pl.Categorical),
                    ],
                )
        df_by_schema[schema_type] = df

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
