import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from seiryo_butterfly_config import ButterflyDiagram
from seiryo_obs_days_config import ObservationsMonthly
from seiryo_sunspot_number_config import (
    SunspotNumberHemispheric,
    SunspotNumberWholeDisk,
)
from seiryo_sunspot_number_with_flare_config import (
    SunspotNumberWithFlare,
    SunspotNumberWithFlareHemispheric,
)
from seiryo_sunspot_number_with_silso_config import (
    SunspotNumberDiff,
    SunspotNumberRatio,
    SunspotNumberRatioDiff1,
    SunspotNumberRatioDiff2,
    SunspotNumberScatter,
    SunspotNumberWithSilso,
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
        pytest.param(
            "config/seiryo/sunspot_number/with_flare.json",
            SunspotNumberWithFlare,
        ),
        pytest.param(
            "config/seiryo/sunspot_number/with_flare_hemispheric.json",
            SunspotNumberWithFlareHemispheric,
        ),
        pytest.param(
            "config/seiryo/sunspot_number/with_silso.json",
            SunspotNumberWithSilso,
        ),
        pytest.param(
            "config/seiryo/sunspot_number/scatter.json", SunspotNumberScatter
        ),
        pytest.param(
            "config/seiryo/sunspot_number/ratio.json", SunspotNumberRatio
        ),
        pytest.param(
            "config/seiryo/sunspot_number/diff.json", SunspotNumberDiff
        ),
        pytest.param(
            "config/seiryo/sunspot_number/ratio_diff_1.json",
            SunspotNumberRatioDiff1,
        ),
        pytest.param(
            "config/seiryo/sunspot_number/ratio_diff_2.json",
            SunspotNumberRatioDiff2,
        ),
        pytest.param(
            "config/seiryo/observations/monthly.json", ObservationsMonthly
        ),
        pytest.param(
            "config/seiryo/butterfly_diagram/monthly.json", ButterflyDiagram
        ),
        pytest.param(
            "config/seiryo/butterfly_diagram/fromtext.json", ButterflyDiagram
        ),
        pytest.param(
            "config/seiryo/butterfly_diagram/merged.json", ButterflyDiagram
        ),
    ],
)
def test_sunspot_number_whole_disk(path: str, config: type[BaseModel]) -> None:
    with Path(path).open("r") as f:
        _ = config(**json.load(f))
