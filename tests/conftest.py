from pathlib import Path

import numpy as np
import pytest
from pint import UnitRegistry


@pytest.fixture()
def sample_smd_file():
    return Path("tests/data/sample.smd")


@pytest.fixture()
def sample_mapping_data() -> "MockMapping":
    class MockMapping:
        def __init__(self):
            self.data = np.random.rand(100, 1000)
            self.step_size = (0.1, 0.1, 0.1)
            self.step_units = ("um", "um", "um")
            self.map_steps = (10, 10, 1)

        def get_spectral_axis(self, channel=0):
            return np.linspace(400, 800, 1000)

        def _get_channel_axis_unit(self, channel=0):
            return "nm"

    return MockMapping()


@pytest.fixture()
def unit_registry() -> UnitRegistry:
    return UnitRegistry()
