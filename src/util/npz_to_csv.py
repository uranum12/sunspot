from pathlib import Path

import numpy as np


def main() -> None:
    out_path = Path("out")

    for path in out_path.rglob("*.npz"):
        with np.load(path) as f:
            for name in f.files:
                file_name = path.with_name(f"{path.stem}_{name}.csv")
                np.savetxt(file_name, f[name], delimiter=",", fmt="%s")


if __name__ == "__main__":
    main()
