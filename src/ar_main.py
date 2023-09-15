from pathlib import Path
from pprint import pprint

import polars as pl

import ar_common
import ar_merge
import ar_new
import ar_notebook
import ar_old


def main() -> None:
    # 入出力先のフォルダのパス
    # 出力先のフォルダがない場合は作成
    data_path = Path("data/fujimori_ar")
    output_path = Path("out/ar")
    output_path.mkdir(parents=True, exist_ok=True)

    # 形式ごとに一つのデータフレームへ結合
    df_notebook = ar_notebook.concat(data_path)
    df_old = ar_old.concat(data_path)
    df_new = ar_new.concat(data_path)

    # 複数のシートに跨って存在するデータを一つに結合
    df_merged = ar_merge.merge(pl.concat([df_old, df_new]))

    # 全ての処理済みを一つのデータフレームへ
    df_all = pl.concat([df_notebook, df_merged])

    # ソート
    df_sorted = ar_common.sort(df_all)

    with pl.StringCache():
        # プロファイルとともに計算
        df, profile = df_sorted.profile()

    # 結果を表示し、保存
    pprint(df.schema)
    print(df)
    print(profile)
    df.write_parquet(output_path / "all.parquet")
    profile.write_csv(output_path / "profile.csv")


if __name__ == "__main__":
    main()
