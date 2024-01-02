from pathlib import Path


def main() -> None:
    ar_path = Path("data/fujimori_ar")
    sn_path = Path("data/fujimori_sn")

    for data_path in [ar_path, sn_path]:
        for file in data_path.glob("* - *.csv"):
            month_dict = {
                "jan": "1",
                "feb": "2",
                "mar": "3",
                "apr": "4",
                "may": "5",
                "jun": "6",
                "jul": "7",
                "aug": "8",
                "sep": "9",
                "oct": "10",
                "nov": "11",
                "dec": "12",
            }
            year, _, month = file.stem.split()
            month = month_dict.get(month, f"err-{month}")
            new_name = f"{year}-{month_dict.get(month, month)}.csv"
            file.rename(file.with_name(new_name))

    for file in Path("data/seiryo").glob("* - *.csv"):
        year, _, b0 = file.stem.split()
        new_name = f"{year}-{b0}.csv"
        file.rename(file.with_name(new_name))


if __name__ == "__main__":
    main()
