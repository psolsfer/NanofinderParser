from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pint import UnitRegistry


@pytest.fixture()
def sample_smd_file() -> Path:
    return Path("tests/data/sample.smd")


class MockMapping:
    def __init__(self) -> None:
        self.data = np.random.rand(100, 1000)
        self.step_size = (0.1, 0.1, 0.1)
        self.step_units = ("um", "um", "um")
        self.map_steps = (10, 10, 1)
        self.map_size = (0.9, 0.9, 0.0)

    def get_spectral_axis(self, channel: int = 0) -> np.ndarray:  # type: ignore[arg-type]
        return np.linspace(400, 800, 1000)

    def _get_channel_axis_unit(self, channel: int = 0) -> str:
        return "nm"


@pytest.fixture()
def sample_mapping_data() -> "MockMapping":
    return MockMapping()


@pytest.fixture()
def unit_registry() -> UnitRegistry:
    return UnitRegistry()
