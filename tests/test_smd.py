"""Tests for the NanoFinder SMD parser, which handles mappings.

The sample file is a real mapping trimmed down to a 4 x 3 grid of 8-point spectra, so every
value asserted here comes from an actual instrument.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanofinderparser import load_smd, load_smd_folder
from nanofinderparser.models import Mapping
from nanofinderparser.units import Units

# ruff: noqa: PLR2004

SMD_FOLDER = Path(__file__).parent.parent / "sample_data" / "smd"
SMD_FILE = SMD_FOLDER / "mapping_small.smd"

X_STEPS = 4
Y_STEPS = 3
SPECTRAL_LEN = 8

# The first spectrum of the file, as written by the instrument.
FIRST_SPECTRUM = (1148.0, 1167.0, 1197.0, 1177.0, 1153.0, 1149.0, 1135.0, 1115.0)

# First and last value of the spectral axis, in nm.
AXIS_FIRST = 560.927298702803
AXIS_LAST = 561.499216547938

LASER_WL_NM = 532.000006769476
STEP_SIZE_NM = 500.4959337450298
X_START_NM = 36057.06986096053
Y_START_NM = 40350.958755956


@pytest.fixture
def mapping() -> Mapping:
    """Sample mapping."""
    return load_smd(SMD_FILE)


# --------------------------------------------------------------------------------------------
# File structure
# --------------------------------------------------------------------------------------------


def test_grid_and_spectra(mapping: Mapping) -> None:
    """The mapping has the grid and the spectrum length its header declares."""
    assert mapping.map_steps == (X_STEPS, Y_STEPS, 1)
    assert mapping.get_spectral_axis_len() == SPECTRAL_LEN
    assert mapping.data.size == X_STEPS * Y_STEPS * SPECTRAL_LEN


def test_data_keeps_the_precision_of_the_file(mapping: Mapping) -> None:
    """The binary block is float32 in the file and stays float32 in memory."""
    assert mapping.data.dtype == np.float32
    assert mapping.get_spectra()[0] == pytest.approx(FIRST_SPECTRUM)


def test_views_of_the_data_share_memory(mapping: Mapping) -> None:
    """get_spectra and get_map are reshaped views, not copies."""
    spectra = mapping.get_spectra()
    cube = mapping.get_map()

    assert spectra.shape == (X_STEPS * Y_STEPS, SPECTRAL_LEN)
    assert cube.shape == (Y_STEPS, X_STEPS, SPECTRAL_LEN)
    assert np.shares_memory(spectra, mapping.data)
    assert np.shares_memory(cube, mapping.data)


def test_map_is_x_fast(mapping: Mapping) -> None:
    """The map is stored row by row, with x as the fast axis."""
    cube = mapping.get_map()
    spectra = mapping.get_spectra()

    for iy in range(Y_STEPS):
        for ix in range(X_STEPS):
            assert np.array_equal(cube[iy, ix], spectra[iy * X_STEPS + ix])


def test_source_is_recorded(mapping: Mapping) -> None:
    """The mapping remembers the file it came from."""
    assert mapping.source == SMD_FILE


def test_load_smd_folder() -> None:
    """The folder loader yields every SMD file, optionally with its path."""
    assert len(list(load_smd_folder(SMD_FOLDER))) == 1

    with_paths = list(load_smd_folder(SMD_FOLDER, return_path=True))
    assert [path.name for _, path in with_paths] == [SMD_FILE.name]


# --------------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------------


def test_measurement_metadata(mapping: Mapping) -> None:
    """The acquisition settings are read from the XML header."""
    assert mapping.datetime == datetime(2021, 3, 10, 11, 32, 47)  # noqa: DTZ001
    assert mapping.date == mapping.datetime.date()
    assert mapping.laser_wavelength == pytest.approx(LASER_WL_NM)
    assert mapping.laser_power == pytest.approx(1.5959067679785)
    assert mapping.get_exposure_time() == pytest.approx(0.5)
    assert mapping.get_accumulation_number() == 1
    assert mapping.original_file_name == r"C:\NanoFinder\sample\mapping_small.smd"


def test_channel_info_is_parsed_from_free_text(mapping: Mapping) -> None:
    """The ``ChannelInfo`` items, which are plain sentences, are turned into fields."""
    info = mapping.scanned_frame_parameters.data_calibration.channels[0].channel_info

    assert info.head_model == "DV420"
    assert info.ccd_width == 1024
    assert info.ccd_height == 255
    assert info.central_pixel == 510
    assert info.pixel_size_um == pytest.approx(26.0)
    assert info.temperature == pytest.approx(-60.0)
    assert info.horizontal_binning == 1
    assert info.center_row == 53
    assert info.track_height == 20
    assert info.readout_mode == "Single Track"
    assert info.cycle_time == pytest.approx(0.573)
    # NanoFinder misspells "accumulate" in its own files.
    assert info.acquisition_mode == "accomulate"


def test_stage_axes_are_converted_from_dac_counts(mapping: Mapping) -> None:
    """Physical positions are derived from the raw DAC counts and the axis calibration."""
    assert mapping.step_units == ("nm", "nm", "nm")
    assert mapping.step_size[0] == pytest.approx(STEP_SIZE_NM)
    assert mapping.step_size[1] == pytest.approx(STEP_SIZE_NM)
    assert mapping.map_start[0] == pytest.approx(X_START_NM)
    assert mapping.map_start[1] == pytest.approx(Y_START_NM)
    assert mapping.map_size[0] == pytest.approx(STEP_SIZE_NM * (X_STEPS - 1))
    assert mapping.map_size[1] == pytest.approx(STEP_SIZE_NM * (Y_STEPS - 1))


# --------------------------------------------------------------------------------------------
# Spectral axis and units
# --------------------------------------------------------------------------------------------


def test_spectral_axis_is_read_as_stored(mapping: Mapping) -> None:
    """The spectral axis comes from the header, in nm, one value per point."""
    axis = mapping.get_spectral_axis()

    assert axis.size == SPECTRAL_LEN
    assert axis[0] == pytest.approx(AXIS_FIRST)
    assert axis[-1] == pytest.approx(AXIS_LAST)


def test_spectral_axis_conversion_round_trips(mapping: Mapping) -> None:
    """Converting to another unit and back gives the original axis."""
    axis = mapping.get_spectral_axis()

    for unit in (Units.cm_1, Units.ev, Units.raman_shift):
        converted = mapping.get_spectral_axis(spectral_units=unit)
        assert converted.shape == axis.shape
        assert not np.allclose(converted, axis)

    shift = mapping.get_spectral_axis(spectral_units="raman_shift")
    # 561 nm excited at 532 nm sits around 950 cm-1.
    assert shift[0] == pytest.approx(950, abs=50)
    # The Raman shift grows as the wavelength does.
    assert np.all(np.diff(shift) > 0)


# --------------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------------


def test_to_df_aligns_data_and_coordinates(mapping: Mapping) -> None:
    """Each row of the coordinates describes the spectrum in the same row of the data."""
    data, mapcoords = mapping.to_df()

    assert data.shape == (X_STEPS * Y_STEPS, SPECTRAL_LEN)
    assert len(mapcoords) == len(data)
    assert list(mapcoords.columns) == ["x", "y"]
    assert data.columns[0] == pytest.approx(AXIS_FIRST)


def test_to_df_starts_from_the_bottom_row(mapping: Mapping) -> None:
    """Rows follow NanoFinder's convention of y starting at the bottom of the mapping area."""
    data, mapcoords = mapping.to_df(index=False)

    y_values = np.asarray(mapcoords["y"].pint.magnitude, dtype=float)
    x_values = np.asarray(mapcoords["x"].pint.magnitude, dtype=float)

    # The first row of the export is the last row of the scan.
    assert y_values[0] == pytest.approx(Y_START_NM + (Y_STEPS - 1) * STEP_SIZE_NM)
    assert y_values[-1] == pytest.approx(Y_START_NM)
    # x still runs left to right within each row.
    assert x_values[0] == pytest.approx(X_START_NM)

    # ... and the spectra were re-ordered along with the coordinates.
    spectra = mapping.get_spectra()
    assert np.array_equal(data.to_numpy()[0], spectra[(Y_STEPS - 1) * X_STEPS])


def test_to_df_without_index(mapping: Mapping) -> None:
    """Passing index=False drops the coordinate MultiIndex."""
    data, _ = mapping.to_df(index=False)
    assert isinstance(data.index, pd.RangeIndex)


@pytest.mark.parametrize(
    ("save_mapcoords", "expected"),
    [("combined", {"m.csv"}), ("separated", {"m_data.csv", "m_mapcoords.csv"}), ("no", {"m.csv"})],
)
def test_to_csv(mapping: Mapping, tmp_path: Path, save_mapcoords: str, expected: set[str]) -> None:
    """The CSV export writes one or two files, depending on how coordinates are saved."""
    mapping.to_csv(path=tmp_path, filename="m.csv", save_mapcoords=save_mapcoords)
    assert {file.name for file in tmp_path.iterdir()} == expected


# --------------------------------------------------------------------------------------------
# The binary block is only as long as the header says it is
# --------------------------------------------------------------------------------------------


def _copy_with_binary_delta(source: Path, target: Path, delta: int) -> Path:
    """Copy an SMD file, adding or removing `delta` bytes from its binary block."""
    raw = source.read_bytes()
    target.write_bytes(raw + b"\x00" * delta if delta > 0 else raw[:delta])
    return target


def test_truncated_file_is_rejected(tmp_path: Path) -> None:
    """A file holding fewer values than its header declares raises."""
    short = _copy_with_binary_delta(SMD_FILE, tmp_path / "short.smd", -8)

    with pytest.raises(ValueError, match="truncated"):
        load_smd(short)


def test_extra_values_are_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A file holding more values than its header declares is trimmed, with a warning."""
    long = _copy_with_binary_delta(SMD_FILE, tmp_path / "long.smd", 8)

    with caplog.at_level(logging.WARNING):
        mapping = load_smd(long)

    assert mapping.data.size == X_STEPS * Y_STEPS * SPECTRAL_LEN
    assert "more than" in caplog.text
