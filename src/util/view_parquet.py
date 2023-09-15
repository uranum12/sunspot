import sys
from pathlib import Path

import polars as pl

if len(sys.argv) > 1:
    path = Path(sys.argv[1])
    print(
        pl.scan_parquet(path)
        .filter(~pl.all_horizontal(pl.all().is_null()))
        .collect()
        .glimpse(return_as_string=True),
    )
