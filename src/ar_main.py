from pathlib import Path
from pprint import pprint

import polars as pl

import ar_common
import ar_merge
import ar_new
import ar_notebook
import ar_old
import ar_type


def main() -> None:
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
        match schema_type := ar_type.detect_schema_type(year, month):
            case ar_type.SchemaType.NOTEBOOK:
                df = ar_notebook.scan_csv(path)
                df = ar_notebook.calc_obs_date(df, year, month)
            case ar_type.SchemaType.OLD:
                df = ar_old.scan_csv(path)
                df = ar_old.calc_obs_date(df, year, month)
            case ar_type.SchemaType.NEW:
                df = ar_new.scan_csv(path)
                df = ar_new.calc_obs_date(df, year, month)
            case _:
                print(f"Err: not supported date for {year}/{month}")
                continue
        dfl_by_schema[schema_type].append(df)

    # 形式ごとに一つのデータフレームへ結合
    df_by_schema: dict[ar_type.SchemaType, pl.LazyFrame] = {}
    for schema_type, dfl in dfl_by_schema.items():
        match schema_type:
            case ar_type.SchemaType.NOTEBOOK:
                df = pl.concat(dfl)
                df = ar_notebook.fill_blanks(df)
                df = ar_common.extract_coords_qm(df, ["lat"])
                df = ar_common.extract_coords_lr(df, ["lat"])
                df = ar_common.extract_coords_sign(df, ["lat"])
                df = ar_common.convert_lat(df)
            case ar_type.SchemaType.OLD | ar_type.SchemaType.NEW:
                df = pl.concat(dfl)
                df = ar_common.extract_no(df)
                df = ar_common.detect_coords_over(df)
                df = ar_common.extract_coords_qm(df)
                df = ar_common.extract_coords_lr(df)
                df = ar_common.extract_coords_sign(df)
                df = ar_common.convert_lat(df)
                df = ar_common.convert_lon(df)
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
                df_by_schema[ar_type.SchemaType.NOTEBOOK],
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
