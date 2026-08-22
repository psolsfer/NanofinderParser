"""Tests for the synthetic mapping generator and the SMD writer.

Everything here is checked by writing a file and reading it back with the parser, so a test
fails either when the generator builds the wrong spectra or when the writer emits a header the
parser cannot make sense of.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from nanofinderparser import (
    BaselineSpec,
    InstrumentSpec,
    MappingSpec,
    MapSpec,
    NoiseSpec,
    PeakSpec,
    SpectralAxisSpec,
    build_mapping,
    build_spectra,
    create_smd,
    load_smd,
)
from nanofinderparser.models import (
    Axis,
    Channel,
    ChannelInfo,
    DataCalibration,
    FrameHeader,
    FrameOptions,
    Mapping,
    ScannedFrameParameters,
    Stage3DParameters,
    StageAxesDimensions,
)
from nanofinderparser.units import Units, convert_spectral_units
from nanofinderparser.write import NIL_GUID

if TYPE_CHECKING:
    from pydantic import BaseModel

SMD_FILE = Path(__file__).parent.parent / "sample_data" / "smd" / "mapping_small.smd"

X_SIZE = 5
Y_SIZE = 4
SPECTRAL_LEN = 64
LASER_NM = 532.0


@pytest.fixture
def spec() -> MappingSpec:
    """Small mapping with a single Gaussian peak and no noise."""
    return MappingSpec(
        map=MapSpec(x_size=X_SIZE, y_size=Y_SIZE, x_step=250.0, y_step=125.0, x_start=1000.0),
        spectral_axis=SpectralAxisSpec(size=SPECTRAL_LEN, start=560.0, stop=600.0),
        peaks=[PeakSpec(center=580.0, fwhm=5.0, amplitude=1000.0)],
        baseline=BaselineSpec(offset=100.0),
        instrument=InstrumentSpec(laser_wavelength_nm=LASER_NM),
    )


# --------------------------------------------------------------------------------------------
# Building the spectra
# --------------------------------------------------------------------------------------------


def test_spectra_have_the_shape_of_the_map(spec: MappingSpec) -> None:
    """The generated cube follows the (slow, fast, spectral) layout of get_map()."""
    spectra = build_spectra(spec)

    assert spectra.shape == (Y_SIZE, X_SIZE, SPECTRAL_LEN)
    assert spectra.dtype == np.float32


def test_peak_sits_where_it_was_asked_to(spec: MappingSpec) -> None:
    """The maximum of every spectrum falls on the center of the peak."""
    axis = spec.spectral_axis.build()
    spectra = build_spectra(spec)

    assert axis[spectra.argmax(axis=-1)] == pytest.approx(580.0, abs=axis[1] - axis[0])


def test_amplitude_and_baseline_are_counts() -> None:
    """The peak rises exactly its amplitude above the baseline."""
    spec = MappingSpec(
        map=MapSpec(x_size=2, y_size=2),
        # An odd number of points over this range puts a sample exactly on the peak.
        spectral_axis=SpectralAxisSpec(size=81, start=560.0, stop=600.0),
        peaks=[PeakSpec(center=580.0, fwhm=5.0, amplitude=1000.0)],
        baseline=BaselineSpec(offset=100.0),
    )

    spectra = build_spectra(spec)

    assert spectra.max() == pytest.approx(1100.0, rel=1e-6)
    assert spectra[..., 0] == pytest.approx(100.0, abs=1.0)


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian", "pseudo_voigt"])
def test_lineshapes_are_half_their_height_at_the_half_width(shape: str) -> None:
    """Whatever the lineshape, the FWHM is the width at half the maximum."""
    spec = MappingSpec(
        map=MapSpec(x_size=1, y_size=1),
        spectral_axis=SpectralAxisSpec(size=2001, start=500.0, stop=600.0),
        peaks=[PeakSpec(center=550.0, fwhm=10.0, amplitude=500.0, shape=shape)],  # type: ignore[arg-type]
    )
    axis = spec.spectral_axis.build()
    spectrum = build_spectra(spec)[0, 0]

    half = np.argmin(np.abs(axis - 555.0))
    assert spectrum[half] == pytest.approx(250.0, rel=1e-2)


def test_unknown_lineshape_is_rejected() -> None:
    """A peak shape the module does not know about raises."""
    spec = MappingSpec(
        map=MapSpec(x_size=2, y_size=2),
        peaks=[PeakSpec(center=550.0, fwhm=5.0, amplitude=1.0, shape="voigt")],  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="Unknown peak shape"):
        build_spectra(spec)


def test_peaks_can_be_given_as_a_raman_shift() -> None:
    """A peak declared in cm-1 lands at the matching wavelength of the stored axis."""
    spec = MappingSpec(
        map=MapSpec(x_size=2, y_size=2),
        spectral_axis=SpectralAxisSpec(size=1024, start=560.0, stop=600.0),
        peaks=[
            PeakSpec(center=1580.0, fwhm=20.0, amplitude=500.0, units=Units.raman_shift),
        ],
        instrument=InstrumentSpec(laser_wavelength_nm=LASER_NM),
    )
    axis = spec.spectral_axis.build()
    expected = convert_spectral_units(
        1580.0, Units.raman_shift, Units.nm, laser_wavelength_nm=LASER_NM
    )

    found = axis[build_spectra(spec).argmax(axis=-1)]

    assert found == pytest.approx(expected, abs=axis[1] - axis[0])


# --------------------------------------------------------------------------------------------
# Parameters changing across the map
# --------------------------------------------------------------------------------------------


def test_a_parameter_can_be_a_function_of_the_coordinates() -> None:
    """A callable amplitude makes the map brighter along x."""
    spec = MappingSpec(
        map=MapSpec(x_size=4, y_size=3, x_step=100.0),
        spectral_axis=SpectralAxisSpec(size=32, start=560.0, stop=600.0),
        peaks=[PeakSpec(center=580.0, fwhm=5.0, amplitude=lambda x, y: 100.0 + x)],
    )

    intensity = build_spectra(spec).max(axis=-1)

    assert np.all(np.diff(intensity, axis=1) > 0)
    assert np.allclose(np.diff(intensity, axis=0), 0.0)


def test_a_parameter_can_be_an_array() -> None:
    """An array parameter is used point by point."""
    amplitudes = np.arange(12, dtype=np.float64).reshape(3, 4) * 100.0 + 100.0
    spec = MappingSpec(
        map=MapSpec(x_size=4, y_size=3),
        spectral_axis=SpectralAxisSpec(size=81, start=560.0, stop=600.0),
        peaks=[PeakSpec(center=580.0, fwhm=10.0, amplitude=amplitudes)],
    )

    intensity = build_spectra(spec).max(axis=-1)

    assert intensity == pytest.approx(amplitudes, rel=1e-5)


def test_a_parameter_of_the_wrong_shape_is_rejected() -> None:
    """An array that does not fit the map raises, naming the offending parameter."""
    spec = MappingSpec(
        map=MapSpec(x_size=4, y_size=3),
        peaks=[PeakSpec(center=580.0, fwhm=5.0, amplitude=np.ones((2, 2)))],
    )

    with pytest.raises(ValueError, match="amplitude of peak 0"):
        build_spectra(spec)


# --------------------------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_mapping(spec: MappingSpec) -> None:
    """Noise is reproducible when a seed is given, and only then."""
    noisy = MappingSpec(
        map=spec.map,
        spectral_axis=spec.spectral_axis,
        peaks=spec.peaks,
        baseline=spec.baseline,
        noise=NoiseSpec(poisson=True, sigma=5.0, seed=42),
    )
    other = MappingSpec(
        map=spec.map,
        spectral_axis=spec.spectral_axis,
        peaks=spec.peaks,
        baseline=spec.baseline,
        noise=NoiseSpec(poisson=True, sigma=5.0, seed=7),
    )

    assert np.array_equal(build_spectra(noisy), build_spectra(noisy))
    assert not np.array_equal(build_spectra(noisy), build_spectra(other))


def test_noise_stays_around_the_noiseless_spectra(spec: MappingSpec) -> None:
    """Poisson noise scatters the counts without shifting their mean."""
    noisy = MappingSpec(
        map=MapSpec(x_size=20, y_size=20),
        spectral_axis=spec.spectral_axis,
        peaks=spec.peaks,
        baseline=spec.baseline,
        noise=NoiseSpec(poisson=True, seed=0),
    )
    clean = MappingSpec(
        map=noisy.map,
        spectral_axis=noisy.spectral_axis,
        peaks=noisy.peaks,
        baseline=noisy.baseline,
    )

    difference = build_spectra(noisy).astype(np.float64) - build_spectra(clean)

    assert np.abs(difference).max() > 0.0
    assert difference.mean() == pytest.approx(0.0, abs=1.0)


# --------------------------------------------------------------------------------------------
# Writing and reading back
# --------------------------------------------------------------------------------------------


def test_a_synthetic_file_reads_back_unchanged(spec: MappingSpec, tmp_path: Path) -> None:
    """The data written to an SMD file is the data the parser reads back."""
    built = build_mapping(spec)
    file = create_smd(tmp_path / "synthetic.smd", spec)

    loaded = load_smd(file)

    assert loaded.map_steps == (X_SIZE, Y_SIZE, 1)
    assert loaded.get_map().shape == (Y_SIZE, X_SIZE, SPECTRAL_LEN)
    assert np.array_equal(loaded.data, built.data)
    assert np.array_equal(loaded.get_spectral_axis(), spec.spectral_axis.build())


def test_the_header_of_a_synthetic_file_survives(spec: MappingSpec, tmp_path: Path) -> None:
    """The acquisition settings of the spec are the ones read back from the file."""
    loaded = load_smd(create_smd(tmp_path / "synthetic.smd", spec))

    assert loaded.laser_wavelength == pytest.approx(LASER_NM)
    assert loaded.step_size[:2] == pytest.approx((250.0, 125.0))
    assert loaded.map_start[:2] == pytest.approx((1000.0, 0.0))
    assert loaded.step_units[0] == "nm"
    assert loaded.datetime == datetime(2024, 1, 1, 12, 0, 0)  # noqa: DTZ001
    assert loaded.get_exposure_time() == pytest.approx(1.0)
    assert loaded.get_accumulation_number() == 1


def test_a_synthetic_file_can_be_exported(spec: MappingSpec, tmp_path: Path) -> None:
    """A synthetic mapping behaves like a parsed one all the way to the csv export."""
    loaded = load_smd(create_smd(tmp_path / "synthetic.smd", spec))

    data, coords = loaded.to_df()

    assert len(coords) == X_SIZE * Y_SIZE
    assert data.shape == (X_SIZE * Y_SIZE, SPECTRAL_LEN)


def test_an_existing_file_can_be_written_back(tmp_path: Path) -> None:
    """A real mapping read from disk survives a write and a second read."""
    original = load_smd(SMD_FILE)
    copy = load_smd(original.to_smd(tmp_path / "copy.smd"))

    assert copy.map_steps == original.map_steps
    assert np.array_equal(copy.data, original.data)
    assert np.array_equal(copy.get_spectral_axis(), original.get_spectral_axis())
    assert copy.datetime == original.datetime
    assert copy.laser_wavelength == pytest.approx(original.laser_wavelength)
    assert copy.step_size == pytest.approx(original.step_size)
    assert copy.map_start == pytest.approx(original.map_start)
    assert copy.get_exposure_time() == pytest.approx(original.get_exposure_time())
    assert copy.get_accumulation_number() == original.get_accumulation_number()
    assert copy.original_file_name == original.original_file_name
    assert copy.single_channel().channel_info == original.single_channel().channel_info


def test_channel_info_items_survive_a_round_trip() -> None:
    """Writing the free-text detector items and reading them back changes nothing."""
    info = ChannelInfo(
        Temperature=-60.0,
        ExposureTime=0.5,
        CycleTime=0.573,
        AcquisitionMode="accomulate",
        AccumulationNumber=3,
        HeadModel="DV420",
        CcdWidth=1024,
        CcdHeight=255,
        CentralPixel=510,
        PixelSizeUm=26.0,
        HorizontalBinning=1,
        CenterRow=53,
        TrackHeight=20,
        ReadoutMode="Single Track",
    )

    items = {f"Item{index}": text for index, text in enumerate(info.to_items())}

    assert Channel.parse_channel_info(items) == info


def test_every_modelled_element_is_written(spec: MappingSpec, tmp_path: Path) -> None:
    """Whatever the models read, the writer writes.

    The models declare the name of every XML element they read; this checks that each of them
    makes it into a written file, so a field added to a model cannot be silently dropped on the
    way out.
    """
    file = create_smd(tmp_path / "synthetic.smd", spec)
    header = file.read_bytes().split(b"</SCANDATA>")[0].decode("utf-8")

    models: list[type[BaseModel]] = [
        Axis,
        Channel,
        DataCalibration,
        FrameHeader,
        FrameOptions,
        ScannedFrameParameters,
        Stage3DParameters,
        StageAxesDimensions,
    ]
    missing = {
        field.alias
        for model in models
        for field in model.model_fields.values()
        if field.alias is not None and f"<{field.alias}>" not in header
    }

    assert not missing


def test_files_written_from_scratch_carry_no_instrument_guids(
    spec: MappingSpec, tmp_path: Path
) -> None:
    """A synthetic file identifies no real hardware."""
    file = create_smd(tmp_path / "synthetic.smd", spec)

    header = file.read_bytes().split(b"</SCANDATA>")[0].decode("utf-8")
    guids = set(re.findall(r"\{[0-9A-Fa-f-]{36}\}", header))

    assert guids == {NIL_GUID}


def test_a_mapping_whose_data_does_not_match_its_header_is_rejected(
    spec: MappingSpec, tmp_path: Path
) -> None:
    """Writing a mapping whose data was tampered with raises instead of writing junk."""
    mapping: Mapping = build_mapping(spec)
    mapping.data = mapping.data[:-1]

    with pytest.raises(ValueError, match="values, but its parameters describe"):
        mapping.to_smd(tmp_path / "broken.smd")
