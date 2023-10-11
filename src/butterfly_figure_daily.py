from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    data_file = Path("out/butterfly/fujimori_daily.npz")

    with np.load(data_file) as f:
        img = f["img"]
        xindex = f["date"]
        yindex = f["lat"]

    xlabel = [
        (i, str(d.year))
        for i, d in enumerate(item.item() for item in xindex)
        if d.day == 1 and d.month == 1 and d.year % 10 == 0
    ]

    ylabel = [(i, n) for i, n in enumerate(yindex) if n % 10 == 0]

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap="binary", aspect=30)

    ax.set_title("butterfly diagram")

    ax.set_xlabel("date")
    ax.set_xticks([i[0] for i in xlabel])
    ax.set_xticklabels([i[1] for i in xlabel])

    ax.set_ylabel("latitude")
    ax.set_yticks([i[0] for i in ylabel])
    ax.set_yticklabels([i[1] for i in ylabel])

    for s in ".pdf", ".png":
        fig.savefig(
            data_file.with_suffix(s),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    plt.show()


if __name__ == "__main__":
    main()
