import sys
from csv import DictReader
from pathlib import Path
from re import compile


def main(argv: list[str]) -> None:
    if len(argv) != 4:  # noqa: PLR2004
        print("Error: Invalid arguments.")
        return

    pattern = compile(
        r"(?P<year>\d{4})"
        r"(?:[-/\. ])"
        r"(?P<month>\d{1,2})"
        r"(?:[-/\. ])"
        r"(?P<day>\d{1,2})"
    )
    year, month, day = map(int, argv[1:])

    for path in Path("data/seiryo").glob("*.csv"):
        with path.open("r") as f:
            lines = f.readlines()
            reader = DictReader(lines)

            match_line_num: list[int] = []
            for row in reader:
                if match := pattern.fullmatch(row["date"]):
                    groups = match.groupdict()
                    if (
                        int(groups["year"]) == year
                        and int(groups["month"]) == month
                        and int(groups["day"]) == day
                    ):
                        match_line_num.append(reader.line_num)

            if len(match_line_num) != 0:
                print(path)
                for line_num in match_line_num:
                    print(f"    {line_num: <4}:{lines[line_num - 1]}", end="")
                print()


if __name__ == "__main__":
    main(sys.argv)
