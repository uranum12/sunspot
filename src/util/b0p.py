import sys
from csv import DictReader
from pathlib import Path


def main(argv: list[str]) -> None:
    if len(argv) != 3:  # noqa: PLR2004
        print("Error: Invalid arguments.")
        return

    month, day = argv[1], argv[2]
    with Path("data/b0p.csv").open("r") as f:
        reader = DictReader(f)
        for row in reader:
            if row["month"] == month and row["day"] == day:
                print(f"B0 : {row['b0']}")
                print(f"P  : {row['p']}")
                return

    print(f"Error: No data for {month}/{day}.")


if __name__ == "__main__":
    main(sys.argv)
