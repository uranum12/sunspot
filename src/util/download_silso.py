from pathlib import Path

import requests


def main() -> None:
    data_path = Path("data")

    base_url = "https://www.sidc.be/SILSO/DATA/"
    file_name = "SN_m_tot_V2.0.txt"

    res = requests.get(base_url + file_name, timeout=20)

    with (data_path / file_name).open("wb") as file:
        file.write(res.content)

    print(f"file {file_name} saved")


if __name__ == "__main__":
    main()
