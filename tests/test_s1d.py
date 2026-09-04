"""Tests for the NanoFinder S1D parser, which handles kinetic series.

The sample file is a real kinetic series trimmed down to 10 spectra of 8 points, so every value
asserted here comes from an actual instrument.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanofinderparser import (
    InstrumentSpec,
    KineticSpec,
    PeakSpec,
    build_kinetic_series,
    create_s1d,
    load_s1d,
    load_s1d_folder,
    load_smd,
    write_s1d,
)
from nanofinderparser.models import Channel, KineticSeries
from nanofinderparser.units import Units

# ruff: noqa: PLR2004

SAMPLE_FOLDER = Path(__file__).parent.parent / "sample_data"
S1D_FOLDER = SAMPLE_FOLDER / "s1d"
S1D_FILE = S1D_FOLDER / "kinetic_small.s1d"
SMD_FILE = SAMPLE_FOLDER / "smd" / "mapping_small.smd"

SERIES_SIZE = 10
SPECTRAL_LEN = 8

# The first and the last spectrum of the file, as written by the instrument.
FIRST_SPECTRUM = (735.0, 737.0, 736.0, 734.0, 735.0, 742.0, 737.0, 739.0)
LAST_SPECTRUM = (1126.0, 1159.0, 1158.0, 1149.0, 1135.0, 1160.0, 1164.0, 1151.0)

# First and last value of the spectral axis, in nm.
AXIS_FIRST = 560.273376464844
AXIS_LAST = 560.845458984375

LASER_WL_NM = 532.0
LASER_POWER_MW = 1.59590673446655

# Timings of the series, as set up on the instrument.
EXPOSURE_TIME = 0.1
ACCUMULATE_CYCLE_TIME = 0.2304
KINETIC_CYCLE_TIME = 0.2304
ACCUMULATIONS = 1

MEASURED_AT = datetime(2026, 9, 3, 16, 30, 24)  # noqa: DTZ001 - the file stores no timezone


@pytest.fixture
def series() -> KineticSeries:
    """Sample kinetic series."""
    return load_s1d(S1D_FILE)


# --------------------------------------------------------------------------------------------
# File structure
# --------------------------------------------------------------------------------------------


def test_series_and_spectra(series: KineticSeries) -> None:
    """The series holds the number of spectra its header declares."""
    assert series.series_size == SERIES_SIZE
    assert len(series) == SERIES_SIZE
    assert series.get_spectral_axis_len() == SPECTRAL_LEN
    assert series.data.size == SERIES_SIZE * SPECTRAL_LEN


def test_data_keeps_the_precision_of_the_file(series: KineticSeries) -> None:
    """The binary block is float32 in the file and stays float32 in memory."""
    assert series.data.dtype == np.float32
    assert series.get_spectra()[0] == pytest.approx(FIRST_SPECTRUM)
    assert series.get_spectra()[-1] == pytest.approx(LAST_SPECTRUM)


def test_spectra_is_a_view_of_the_data(series: KineticSeries) -> None:
    """get_spectra is a reshaped view, not a copy."""
    spectra = series.get_spectra()

    assert spectra.shape == (SERIES_SIZE, SPECTRAL_LEN)
    assert np.shares_memory(spectra, series.data)


def test_spectral_axis(series: KineticSeries) -> None:
    """The spectral axis is shared by every spectrum and is stored in nm."""
    axis = series.get_spectral_axis()

    assert axis.size == SPECTRAL_LEN
    assert axis[0] == pytest.approx(AXIS_FIRST)
    assert axis[-1] == pytest.approx(AXIS_LAST)
    assert series._get_channel_axis_unit() == "nm"  # noqa: SLF001


def test_spectral_axis_conversion(series: KineticSeries) -> None:
    """The spectral axis converts to other units, as it does for a mapping."""
    in_nm = series.get_spectral_axis()
    shifted = series.get_spectral_axis(Units.raman_shift)

    assert shifted.size == in_nm.size
    # A longer wavelength than the laser is a positive Stokes shift, growing with wavelength.
    assert np.all(shifted > 0)
    assert np.all(np.diff(shifted) > 0)


def test_source_is_recorded(series: KineticSeries) -> None:
    """The series remembers the file it came from."""
    assert series.source == S1D_FILE


def test_instrument_settings(series: KineticSeries) -> None:
    """The laser and the acquisition settings come from the header."""
    assert series.laser_wavelength == pytest.approx(LASER_WL_NM)
    assert series.laser_power == pytest.approx(LASER_POWER_MW)
    assert series.datetime == MEASURED_AT
    assert series.date == MEASURED_AT.date()


def test_load_s1d_folder() -> None:
    """The folder loader finds the sample and reports its path."""
    loaded = list(load_s1d_folder(S1D_FOLDER, return_path=True))

    assert [path for _, path in loaded] == [S1D_FILE]
    assert all(item.series_size == SERIES_SIZE for item, _ in loaded)


# --------------------------------------------------------------------------------------------
# The two formats do not get mixed up
# --------------------------------------------------------------------------------------------


def test_load_s1d_rejects_an_smd_file() -> None:
    """Reading a mapping as a kinetic series says so, and points at the right function."""
    with pytest.raises(ValueError, match="load_smd"):
        load_s1d(SMD_FILE)


def test_load_smd_rejects_an_s1d_file() -> None:
    """Reading a kinetic series as a mapping says so, and points at the right function."""
    with pytest.raises(ValueError, match="load_s1d"):
        load_smd(S1D_FILE)


# --------------------------------------------------------------------------------------------
# Timings
# --------------------------------------------------------------------------------------------


def test_timings_are_read(series: KineticSeries) -> None:
    """The free-text items of the header carry the timings of the kinetic series."""
    assert series.exposure_time == pytest.approx(EXPOSURE_TIME)
    assert series.accumulate_cycle_time == pytest.approx(ACCUMULATE_CYCLE_TIME)
    assert series.kinetic_cycle_time == pytest.approx(KINETIC_CYCLE_TIME)
    assert series.accumulation_number == ACCUMULATIONS


def test_time_axis(series: KineticSeries) -> None:
    """The time axis starts at zero and is spaced by the kinetic cycle time."""
    times = series.time_axis

    assert times.size == SERIES_SIZE
    assert times[0] == 0.0
    assert np.diff(times) == pytest.approx(KINETIC_CYCLE_TIME)
    assert series.time_step == pytest.approx(KINETIC_CYCLE_TIME)
    assert series.duration == pytest.approx((SERIES_SIZE - 1) * KINETIC_CYCLE_TIME)


def test_timestamps(series: KineticSeries) -> None:
    """The timestamps offset the start of the measurement by the time axis."""
    stamps = series.timestamps

    assert len(stamps) == SERIES_SIZE
    assert stamps[0] == MEASURED_AT
    assert stamps[-1] == MEASURED_AT + timedelta(seconds=(SERIES_SIZE - 1) * KINETIC_CYCLE_TIME)


def test_time_step_falls_back_to_the_cycle_time(series: KineticSeries) -> None:
    """A file without a kinetic cycle time uses the accumulate cycle time instead."""
    info = series.single_channel().channel_info
    info.kinetic_cycle_time = None

    assert series.time_step == pytest.approx(ACCUMULATE_CYCLE_TIME)


def test_time_axis_without_any_timing(series: KineticSeries) -> None:
    """A file that records no timing cannot produce a time axis."""
    info = series.single_channel().channel_info
    info.kinetic_cycle_time = None
    info.cycle_time = None
    info.exposure_time = None

    assert series.time_step is None
    with pytest.raises(ValueError, match="unknown"):
        _ = series.time_axis


# --------------------------------------------------------------------------------------------
# The free-text ChannelInfo items
# --------------------------------------------------------------------------------------------


KINETIC_ACQUISITION_ITEM = (
    "Acquisition mode: Kinetic series. Number in Accumulation = 3  Number in Kinetic series = 500"
)
KINETIC_EXPOSURE_ITEM = (
    "Exposure time, [sec] = 0.1000 Cycle time [sec] = 0.2304 Kinetic Cycle time [sec] = 1.5180"
)
ACCUMULATE_ACQUISITION_ITEM = "Acquisition mode: Accomulate. Number in Accumulation = 4"
ACCUMULATE_EXPOSURE_ITEM = "Exposure time, [sec] = 1.0000 Cycle time [sec] = 1.2000"


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        (
            [KINETIC_ACQUISITION_ITEM, KINETIC_EXPOSURE_ITEM],
            {
                "acquisition_mode": "kinetic series",
                "accumulation_number": 3,
                "kinetic_series_number": 500,
                "exposure_time": 0.1,
                "cycle_time": 0.2304,
                "kinetic_cycle_time": 1.518,
            },
        ),
        (
            [ACCUMULATE_ACQUISITION_ITEM, ACCUMULATE_EXPOSURE_ITEM],
            {
                "acquisition_mode": "accomulate",
                "accumulation_number": 4,
                "kinetic_series_number": None,
                "exposure_time": 1.0,
                "cycle_time": 1.2,
                "kinetic_cycle_time": None,
            },
        ),
    ],
    ids=["kinetic", "accumulate"],
)
def test_channel_info_reads_the_free_text_items(
    items: list[str], expected: dict[str, object]
) -> None:
    """Both the kinetic and the accumulated wording of the items are understood."""
    channel = Channel(
        DeviceGuid="",
        DeviceName="Andor CCD",
        DataChannelName="Photons",
        DataChannelUnit="Counts",
        ChannelSize=2,
        ChannelAxisName="Wavelength",
        ChannelAxisUnit="nm",
        ChannelAxisLaserWl=532.0,
        ChannelAxisArray="500.0 501.0",
        SeriesSize=1,
        ChannelInfo={f"Item{index}": text for index, text in enumerate(items)},
    )
    info = channel.channel_info

    for attribute, value in expected.items():
        found = getattr(info, attribute)
        if isinstance(value, float):
            assert found == pytest.approx(value)
        else:
            assert found == value


def test_channel_info_writes_the_items_back(series: KineticSeries) -> None:
    """The items of a kinetic series survive a read-write cycle unchanged."""
    info = series.single_channel().channel_info
    items = info.to_items()

    acquisition = next(item for item in items if item.startswith("Acquisition mode"))
    exposure = next(item for item in items if item.startswith("Exposure time"))

    assert acquisition == (
        "Acquisition mode: Kinetic series. Number in Accumulation = 1  "
        f"Number in Kinetic series = {SERIES_SIZE}"
    )
    assert exposure == (
        "Exposure time, [sec] = 0.1000 Cycle time [sec] = 0.2304 Kinetic Cycle time [sec] = 0.2304"
    )


# --------------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------------


def test_to_df_is_indexed_by_time(series: KineticSeries) -> None:
    """Each row of the DataFrame is a spectrum, indexed by its nominal time."""
    frame = series.to_df()

    assert frame.shape == (SERIES_SIZE, SPECTRAL_LEN)
    assert frame.index.name == "time (s)"
    assert frame.index[1] == pytest.approx(KINETIC_CYCLE_TIME)
    assert frame.columns[0] == pytest.approx(AXIS_FIRST)
    assert frame.iloc[0].to_numpy() == pytest.approx(FIRST_SPECTRUM)


def test_to_df_can_be_indexed_by_series(series: KineticSeries) -> None:
    """The rows can also be numbered rather than timed."""
    frame = series.to_df(index="series")

    assert frame.index.name == "series"
    assert list(frame.index) == list(range(SERIES_SIZE))


def test_to_csv(series: KineticSeries, tmp_path: Path) -> None:
    """The csv holds the same numbers as the DataFrame."""
    written = series.to_csv(tmp_path)

    assert written == tmp_path / "kinetic_small.csv"

    read_back = pd.read_csv(written, index_col=0)
    assert read_back.shape == (SERIES_SIZE, SPECTRAL_LEN)
    assert read_back.to_numpy()[0] == pytest.approx(FIRST_SPECTRUM)


# --------------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------------


def test_write_s1d_round_trip(series: KineticSeries, tmp_path: Path) -> None:
    """A series written back and read again holds the same data and settings."""
    copy = load_s1d(write_s1d(series, tmp_path / "copy.s1d"))

    assert np.array_equal(copy.data, series.data)
    assert np.array_equal(copy.get_spectral_axis(), series.get_spectral_axis())
    assert copy.series_size == series.series_size
    assert copy.datetime == series.datetime
    assert copy.laser_wavelength == pytest.approx(series.laser_wavelength)
    assert copy.exposure_time == pytest.approx(series.exposure_time)
    assert copy.kinetic_cycle_time == pytest.approx(series.kinetic_cycle_time)
    assert copy.accumulation_number == series.accumulation_number


def test_to_s1d_is_the_method_form_of_write_s1d(series: KineticSeries, tmp_path: Path) -> None:
    """The method on the model writes the same file the function does."""
    from_method = series.to_s1d(tmp_path / "method.s1d")
    from_function = write_s1d(series, tmp_path / "function.s1d")

    assert from_method.read_bytes() == from_function.read_bytes()


def test_writing_rejects_inconsistent_data(series: KineticSeries, tmp_path: Path) -> None:
    """A series whose data does not match its header is not written."""
    series.data = series.data[:-1]

    with pytest.raises(ValueError, match="values"):
        write_s1d(series, tmp_path / "broken.s1d")


# --------------------------------------------------------------------------------------------
# Malformed files
# --------------------------------------------------------------------------------------------


def test_truncated_file_is_rejected(tmp_path: Path) -> None:
    """A file whose binary block is short of what its header declares raises."""
    truncated = tmp_path / "truncated.s1d"
    truncated.write_bytes(S1D_FILE.read_bytes()[:-8])

    with pytest.raises(ValueError, match="truncated"):
        load_s1d(truncated)


def test_extra_values_are_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A file holding more values than declared is truncated, with a warning."""
    padded = tmp_path / "padded.s1d"
    padded.write_bytes(S1D_FILE.read_bytes() + b"\x00" * 8)

    with caplog.at_level(logging.WARNING):
        loaded = load_s1d(padded)

    assert loaded.data.size == SERIES_SIZE * SPECTRAL_LEN
    assert "more than" in caplog.text


# --------------------------------------------------------------------------------------------
# Synthetic series
# --------------------------------------------------------------------------------------------


def test_build_kinetic_series() -> None:
    """A synthetic series has the size and the timings of its spec."""
    spec = KineticSpec(
        series_size=6,
        instrument=InstrumentSpec(exposure_time_s=0.4, kinetic_cycle_time_s=0.5),
    )
    built = build_kinetic_series(spec)

    assert built.series_size == 6
    assert built.get_spectra().shape == (6, spec.spectral_axis.size)
    assert built.kinetic_cycle_time == pytest.approx(0.5)
    assert built.time_axis == pytest.approx(spec.times())


def test_synthetic_peaks_evolve_with_time() -> None:
    """A peak parameter given as a function of time changes along the series."""
    spec = KineticSpec(
        series_size=5,
        peaks=[PeakSpec(center=550.0, fwhm=2.0, amplitude=lambda t, _: 100.0 + 100.0 * t)],
        instrument=InstrumentSpec(exposure_time_s=1.0, kinetic_cycle_time_s=1.0),
    )
    spectra = build_kinetic_series(spec).get_spectra()
    maxima = spectra.max(axis=1)

    assert np.all(np.diff(maxima) > 0)


def test_create_s1d_round_trip(tmp_path: Path) -> None:
    """A synthetic series written to a file reads back unchanged."""
    spec = KineticSpec(
        series_size=4,
        peaks=[PeakSpec(center=550.0, fwhm=2.0, amplitude=500.0)],
        instrument=InstrumentSpec(exposure_time_s=0.2, kinetic_cycle_time_s=0.25),
    )
    built = build_kinetic_series(spec)
    loaded = load_s1d(create_s1d(tmp_path / "synthetic.s1d", spec))

    assert np.array_equal(loaded.data, built.data)
    assert loaded.series_size == 4
    assert loaded.kinetic_cycle_time == pytest.approx(0.25)
    assert loaded.time_axis == pytest.approx([0.0, 0.25, 0.5, 0.75])
