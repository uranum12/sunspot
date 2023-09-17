from pathlib import Path
from pprint import pprint

import polars as pl

import sn_common
import sn_jst
import sn_type
import sn_ut


def main() -> None:
    # 入出力先のフォルダのパス
    # 出力先のフォルダがない場合は作成
    data_path = Path("data/fujimori_sn")
    output_path = Path("out/sn")
    output_path.mkdir(parents=True, exist_ok=True)

    file_frames: list[pl.LazyFrame] = []
    for path in data_path.glob("*-*.csv"):
        # ファイル名から対象の年と月を計算
        year, month = map(int, path.stem.split("-"))
        file_frame = pl.scan_csv(
            path,
            dtypes={
                "date": pl.UInt8,
                "time": pl.Utf8,
                "ng": pl.UInt8,
                "nf": pl.UInt16,
                "sg": pl.UInt8,
                "sf": pl.UInt16,
                "remarks": pl.Utf8,
            },
        )
        # jstとutで時刻の計算方法が別
        match sn_type.detect_time_type(year, month):
            case sn_type.TimeType.JST:
                file_frame = sn_jst.calc_time(file_frame)
            case sn_type.TimeType.UT:
                file_frame = sn_ut.calc_time(file_frame)
            case _:
                print(f"Err: not supported date for {year}/{month}")
                continue
        # 日付の計算
        file_frame = sn_common.calc_date(file_frame, year, month)
        file_frames.append(file_frame)
    # 全てのファイルを一つへ結合
    all_files = pl.concat(file_frames)

    # ソートし、プロファイルとともに保存
    df, profile = sn_common.sort(all_files).profile()

    # 結果を表示し、保存
    pprint(df.schema)
    print(df)
    print(profile)
    df.write_parquet(output_path / "all.parquet")
    profile.write_csv(output_path / "profile.csv")


if __name__ == "__main__":
    main()
