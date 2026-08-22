"""Tests for the catalog of synthetic samples.

Every sample is built and measured the way a user would measure it: a band is integrated over a
window of the spectral axis and its center of mass compared with the position the sample claims
to put it at. A test fails either when a sample is described wrongly or when the generator stops
putting the peaks where the spec asks for them.
"""

from pathlib import Path
from typing import get_args

import numpy as np
import pytest
from numpy.typing import NDArray
from typeguard import suppress_type_checks

from nanofinderparser import create_smd, load_smd
from nanofinderparser.models import Mapping
from nanofinderparser.samples import (
    BUILDERS,
    SAMPLES,
    SampleName,
    sample_mapping,
    sample_spec,
)

X_SIZE = 12
Y_SIZE = 9
SPECTRAL_LEN = 1024

# Window around the A exciton of MoS2, in nm.
A_EXCITON_NM = (650.0, 700.0)

SAMPLE_NAMES: tuple[SampleName, ...] = get_args(SampleName.__value__)


def build(name: SampleName) -> Mapping:
    """Build a small mapping of one of the samples.

    Parameters
    ----------
    name : SampleName
        Which sample to build.

    Returns
    -------
    Mapping
        The mapping, small enough to keep the tests quick.
    """
    return sample_mapping(name, x_size=X_SIZE, y_size=Y_SIZE, n_points=SPECTRAL_LEN)


def band_center(mapping: Mapping, low: float, high: float) -> NDArray[np.float64]:
    """Center of mass of a band, at every point of the map.

    Parameters
    ----------
    mapping : Mapping
        The mapping to measure.
    low, high : float
        Limits of the band, in cm-1.

    Returns
    -------
    NDArray[np.float64]
        The position of the band, in cm-1, of the shape of the map.
    """
    shift = mapping.get_spectral_axis("raman_shift")
    inside = (shift >= low) & (shift <= high)
    axis = shift[inside]
    band = mapping.get_map()[:, :, inside].astype(np.float64)
    above = np.clip(band - np.median(band, axis=-1, keepdims=True), 0.0, None)
    center: NDArray[np.float64] = (above * axis).sum(axis=-1) / above.sum(axis=-1)
    return center


def band_height(mapping: Mapping, low: float, high: float) -> NDArray[np.float64]:
    """Height of a band above its own background, at every point of the map.

    Parameters
    ----------
    mapping : Mapping
        The mapping to measure.
    low, high : float
        Limits of the band, in cm-1.

    Returns
    -------
    NDArray[np.float64]
        The height of the band, in counts, of the shape of the map.
    """
    shift = mapping.get_spectral_axis("raman_shift")
    inside = (shift >= low) & (shift <= high)
    band = mapping.get_map()[:, :, inside].astype(np.float64)
    height: NDArray[np.float64] = (band - np.median(band, axis=-1, keepdims=True)).max(axis=-1)
    return height


def test_catalog_is_complete() -> None:
    """Every name of the alias has both a builder and a description, and nothing else does."""
    assert set(BUILDERS) == set(SAMPLE_NAMES)
    assert set(SAMPLES) == set(SAMPLE_NAMES)
    assert all(info.name == name for name, info in SAMPLES.items())


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_sample_builds(name: SampleName) -> None:
    """Every sample builds a mapping of the requested size, holding plausible counts."""
    mapping = build(name)
    cube = mapping.get_map()

    assert cube.shape == (Y_SIZE, X_SIZE, SPECTRAL_LEN)
    assert np.isfinite(cube).all()
    assert cube.min() >= 0.0
    assert mapping.map_steps == (X_SIZE, Y_SIZE, 1)
    assert mapping.laser_wavelength == pytest.approx(532.0)


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_sample_is_reproducible(name: SampleName) -> None:
    """The seed of the catalog makes a sample come out the same every time."""
    assert np.array_equal(build(name).get_map(), build(name).get_map())


def test_unknown_sample_is_rejected() -> None:
    """Asking for a sample that is not in the catalog says so, and lists the ones that are.

    Type checking is suppressed because the test calls the function the way unchecked code
    would, which is precisely the case the error message is written for.
    """
    message = r"Unknown sample 'gold'.*graphene"
    with suppress_type_checks(), pytest.raises(ValueError, match=message):
        sample_spec("gold")  # type: ignore[arg-type]


def test_graphene_bands() -> None:
    """Graphene softens along x, and its 2D band dims over the bilayer patch."""
    mapping = build("graphene")

    g_position = band_center(mapping, 1500.0, 1700.0)
    # The G band is specified to soften from 1584 to 1578 cm-1 from left to right.
    assert g_position[:, 0].mean() == pytest.approx(1584.0, abs=1.5)
    assert g_position[:, -1].mean() == pytest.approx(1578.0, abs=1.5)

    two_d = band_height(mapping, 2550.0, 2850.0)
    assert two_d[Y_SIZE // 2, X_SIZE // 2] < 0.5 * two_d[0, 0]

    d_band = band_height(mapping, 1250.0, 1450.0)
    assert d_band[:, 0].mean() > 5.0 * d_band[:, -1].mean()


def test_silicon_line_is_where_it_should_be() -> None:
    """The silicon line sits at 520.7 cm-1, save for the relaxed patch."""
    position = band_center(build("silicon"), 480.0, 560.0)

    assert position[0, 0] == pytest.approx(520.7, abs=0.3)
    assert position.min() == pytest.approx(519.6, abs=0.3)


def test_hbn_mode_shifts_over_the_thin_region() -> None:
    """The E2g mode of hBN creeps up by a few wavenumbers where the flake is thinner."""
    position = band_center(build("hbn"), 1300.0, 1430.0)

    assert position.min() == pytest.approx(1366.0, abs=1.0)
    assert position.max() > position.min() + 2.0


def test_mos2_exciton_is_quenched_over_the_bilayer() -> None:
    """The A exciton of MoS2 loses most of its height over the bilayer patch."""
    mapping = build("mos2")
    axis = mapping.get_spectral_axis("nm")
    cube = mapping.get_map().astype(np.float64)

    inside = (axis >= A_EXCITON_NM[0]) & (axis <= A_EXCITON_NM[1])
    height = (cube[:, :, inside] - np.median(cube, axis=-1, keepdims=True)).max(axis=-1)

    monolayer = height[0, 0]
    bilayer = height[int(0.55 * Y_SIZE), int(0.42 * X_SIZE)]
    substrate = height[:, -1].mean()

    assert bilayer < 0.5 * monolayer
    assert substrate < 0.2 * monolayer


def test_noise_sample_holds_no_peaks() -> None:
    """The background-only sample is flat enough that nothing stands out of it."""
    spec = sample_spec("noise")
    assert list(spec.peaks) == []

    cube = build("noise").get_map().astype(np.float64)
    spread = cube.std(axis=-1)
    height = cube.max(axis=-1) - np.median(cube, axis=-1)
    # Only the tail of the noise pokes out of the background, never a peak.
    assert (height < 8.0 * spread).all()


def test_sample_survives_a_round_trip(tmp_path: Path) -> None:
    """A sample written to an SMD file comes back unchanged."""
    written = sample_mapping("graphene", x_size=5, y_size=4, n_points=128)
    file = create_smd(
        tmp_path / "graphene.smd", sample_spec("graphene", x_size=5, y_size=4, n_points=128)
    )

    read_back = load_smd(file)

    assert np.array_equal(read_back.get_map(), written.get_map())
    assert read_back.map_steps == written.map_steps
