import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from seiryo_sunspot_number import (
    SunspotNumberHemispheric,
    SunspotNumberWholeDisk,
)


@pytest.mark.parametrize(
    ("path", "config"),
    [
        pytest.param(
            "config/seiryo/sunspot_number/whole_disk.json",
            SunspotNumberWholeDisk,
        ),
        pytest.param(
            "config/seiryo/sunspot_number/hemispheric.json",
            SunspotNumberHemispheric,
        ),
    ],
)
def test_sunspot_number_whole_disk(path: str, config: type[BaseModel]) -> None:
    with Path(path).open("r") as f:
        _ = config(**json.load(f))
