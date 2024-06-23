from pathlib import Path

import requests


def main() -> None:
    data_path = Path("data/flare")

    for year in range(2011, 2023 + 1):
        for hemisphere in ["total", "north", "south"]:
            base_url = (
                "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
                f"solar-features/solar-flares/index/flare-index/{year}/"
            )
            file_name = f"flare-index-{hemisphere}_{year}.txt"

            res = requests.get(base_url + file_name, timeout=20)

            with (data_path / file_name).open("wb") as file:
                file.write(res.content)

            print(f"file {file_name} saved")


if __name__ == "__main__":
    main()
