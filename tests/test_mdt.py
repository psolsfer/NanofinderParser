"""Tests for the NT-MDT (".mdt") parser, which handles individual spectra and 2-D maps."""

import logging
import struct
from pathlib import Path

import numpy as np
import pytest

from nanofinderparser import load_mdt, load_mdt_folder, load_mdt_images
from nanofinderparser.models import Image, Images, Spectra, Spectrum
from nanofinderparser.parsers import (
    MdtImageFrame,
    MdtSpectrumFrame,
    _read_mdt_value_counts,
    read_mdt_frames,
)
from nanofinderparser.units import MdtUnit, Units

# ruff: noqa: PLR2004

MDT_FOLDER = Path(__file__).parent.parent / "sample_data" / "mdt"

SPECTRA_FILE = "Spectra.mdt"
MIXED_FILE = "Spectra_and_2DMaps.mdt"

# Expected (number of spectra, number of maps) for every sample file.
FILE_CONTENTS: dict[str, tuple[int, int]] = {
    SPECTRA_FILE: (2, 0),
    MIXED_FILE: (2, 1),
}


@pytest.fixture
def spectra() -> Spectra:
    """Spectra of the spectra-only sample file."""
    return load_mdt(MDT_FOLDER / SPECTRA_FILE)


@pytest.fixture
def maps() -> Images:
    """2-D maps of the mixed sample file."""
    return load_mdt_images(MDT_FOLDER / MIXED_FILE)


# --------------------------------------------------------------------------------------------
# File structure
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "expected"), FILE_CONTENTS.items())
def test_file_contents(name: str, expected: tuple[int, int]) -> None:
    """Every sample file yields the expected number of spectra and maps."""
    assert (len(load_mdt(MDT_FOLDER / name)), len(load_mdt_images(MDT_FOLDER / name))) == expected


@pytest.mark.parametrize("name", FILE_CONTENTS)
def test_every_frame_is_typed_and_titled(name: str) -> None:
    """All frames decode into a known kind and carry a non-empty title."""
    frames = read_mdt_frames(MDT_FOLDER / name)
    assert frames
    for frame in frames:
        assert isinstance(frame, MdtSpectrumFrame | MdtImageFrame)
        assert frame.title


def test_mixed_file_splits_spectra_and_maps() -> None:
    """A file holding both kinds exposes each through its own loader, losing nothing."""
    path = MDT_FOLDER / MIXED_FILE
    assert len(read_mdt_frames(path)) == len(load_mdt(path)) + len(load_mdt_images(path))


def test_load_mdt_folder() -> None:
    """The folder loader yields every MDT file, optionally with its path."""
    assert len(list(load_mdt_folder(MDT_FOLDER))) == len(FILE_CONTENTS)

    with_paths = list(load_mdt_folder(MDT_FOLDER, return_path=True))
    assert {path.name for _, path in with_paths} == set(FILE_CONTENTS)


# --------------------------------------------------------------------------------------------
# Value counts, which NanoFinder stores twice and does not always fill in
# --------------------------------------------------------------------------------------------

VAR_SIZE = 442
SPECTRA_LEN = 1024


def _body_declaring(vars_counts: tuple[int, int], data_counts: tuple[int, int]) -> bytes:
    """Build a minimal frame body declaring the given value counts in both places."""
    body = bytearray(VAR_SIZE + 8)
    struct.pack_into("<HH", body, 32, *vars_counts)
    struct.pack_into("<HH", body, VAR_SIZE + 2, *data_counts)
    return bytes(body)


def test_value_counts_come_from_the_data_header() -> None:
    """The counts stored next to the data are used, even when the variables are left empty."""
    assert _read_mdt_value_counts(_body_declaring((0, 0), (SPECTRA_LEN, 2)), VAR_SIZE, 0) == (
        SPECTRA_LEN,
        2,
    )


def test_value_counts_fall_back_to_the_frame_variables() -> None:
    """The counts in the variables are used when the data header is left empty."""
    assert _read_mdt_value_counts(_body_declaring((SPECTRA_LEN, 2), (0, 0)), VAR_SIZE, 0) == (
        SPECTRA_LEN,
        2,
    )


def test_missing_value_counts_are_rejected() -> None:
    """A frame that declares no counts at all is rejected."""
    with pytest.raises(ValueError, match="does not declare how many values"):
        _read_mdt_value_counts(_body_declaring((0, 0), (0, 0)), VAR_SIZE, 0)


def test_conflicting_value_counts_are_rejected() -> None:
    """A frame whose two copies of the counts disagree is rejected."""
    with pytest.raises(ValueError, match="Inconsistent MDT frame"):
        _read_mdt_value_counts(_body_declaring((512, 2), (SPECTRA_LEN, 2)), VAR_SIZE, 0)


def test_spectra_parse_despite_empty_frame_variables() -> None:
    """Real files leave the counts in the variables at zero, and must still parse.

    This guards the fallback above against a regression that would break these sample files.
    """
    frames = read_mdt_frames(MDT_FOLDER / SPECTRA_FILE)
    for frame in frames:
        assert isinstance(frame, MdtSpectrumFrame)
        assert (frame.point_count, frame.array_count) == (SPECTRA_LEN, 2)
        assert frame.arrays.shape == (2, SPECTRA_LEN)


# --------------------------------------------------------------------------------------------
# Spectra
# --------------------------------------------------------------------------------------------


def test_spectrum_metadata(spectra: Spectra) -> None:
    """The metadata of a spectrum is read correctly."""
    spectrum = spectra[0]
    assert isinstance(spectrum, Spectrum)
    assert spectrum.title == "Spectrum_1"
    assert spectrum.laser_wavelength == 532.0
    assert spectrum.spectral_axis_unit is Units.nm
    assert spectrum.data_unit == "counts"
    assert spectrum.spectral_axis_len == SPECTRA_LEN
    assert spectrum.datetime is not None
    assert spectrum.date == spectrum.datetime.date()


def test_spectral_axis_is_a_well_formed_wavelength_axis(spectra: Spectra) -> None:
    """The stored axis is finite, strictly increasing, and within an optical range."""
    for spectrum in spectra:
        axis = spectrum.spectral_axis
        assert axis.shape == spectrum.data.shape
        assert np.all(np.isfinite(axis))
        assert np.all(np.diff(axis) > 0)
        assert 200.0 < axis[0] < axis[-1] < 2000.0


def test_intensities_are_whole_counts(spectra: Spectra) -> None:
    """Intensities decode to whole photon counts.

    Counts are integers by nature, so recovering them exactly is a strong check that the stored
    float32 values are being read with the right offset, size, and byte order.
    """
    for spectrum in spectra:
        assert np.all(spectrum.data >= 0)
        assert spectrum.data == pytest.approx(np.round(spectrum.data))


def test_zero_raman_shift_falls_at_the_laser_wavelength(spectra: Spectra) -> None:
    """Converting to Raman shift puts the origin at the excitation wavelength.

    A pure property of the conversion, so it holds whatever was measured. It ties together the
    spectral axis and the laser wavelength read from the file.
    """
    spanning = [
        spectrum
        for spectrum in spectra
        if spectrum.laser_wavelength is not None
        and spectrum.spectral_axis[0] < spectrum.laser_wavelength < spectrum.spectral_axis[-1]
    ]
    assert spanning, "no sample spectrum covers its own excitation wavelength"

    for spectrum in spanning:
        shift = spectrum.get_spectral_axis("raman_shift")
        crossing = int(np.argmin(np.abs(shift)))
        assert spectrum.spectral_axis[crossing] == pytest.approx(spectrum.laser_wavelength, abs=0.1)


def test_excitation_line_is_the_strongest_feature(spectra: Spectra) -> None:
    """In a sample spectrum that covers it, the excitation line dominates.

    The elastically scattered laser light is orders of magnitude stronger than any inelastic
    signal, so this checks that the axis and the intensities are aligned with each other, and
    not shifted or reversed.
    """
    spectrum = spectra["Spectrum_2"]
    shift = spectrum.get_spectral_axis("raman_shift")
    assert abs(shift[spectrum.data.argmax()]) < 5.0


def test_get_spectral_axis_defaults_to_stored_units(spectra: Spectra) -> None:
    """Asking for no units, or for the stored units, returns the untouched axis."""
    spectrum = spectra[0]
    assert np.array_equal(spectrum.get_spectral_axis(), spectrum.spectral_axis)
    assert np.array_equal(spectrum.get_spectral_axis("nm"), spectrum.spectral_axis)


@pytest.mark.parametrize("units", [Units.raman_shift, Units.cm_1, Units.ev])
def test_spectral_axis_conversion_round_trip(spectra: Spectra, units: Units) -> None:
    """Converting the axis away from nm and back recovers the original values."""
    spectrum = spectra[0]
    converted = spectrum.get_spectral_axis(units)
    back = Spectrum(
        title=spectrum.title,
        spectral_axis=converted,
        data=spectrum.data,
        spectral_axis_unit=units,
        data_unit=spectrum.data_unit,
        laser_wavelength=spectrum.laser_wavelength,
        measured_at=spectrum.measured_at,
    ).get_spectral_axis("nm")
    assert back == pytest.approx(spectrum.spectral_axis, rel=1e-9)


def test_raman_shift_needs_the_laser_wavelength(spectra: Spectra) -> None:
    """Converting to Raman shift without a recorded excitation wavelength is refused."""
    spectrum = spectra[0]
    without_laser = Spectrum(
        title=spectrum.title,
        spectral_axis=spectrum.spectral_axis,
        data=spectrum.data,
        spectral_axis_unit=spectrum.spectral_axis_unit,
        data_unit=spectrum.data_unit,
        laser_wavelength=None,
        measured_at=None,
    )
    with pytest.raises(ValueError, match="excitation wavelength"):
        without_laser.get_spectral_axis("raman_shift")
    # A conversion that does not involve the laser still works.
    assert without_laser.get_spectral_axis("eV").shape == spectrum.spectral_axis.shape


# --------------------------------------------------------------------------------------------
# Collections
# --------------------------------------------------------------------------------------------


def test_lookup_by_title_and_slicing(spectra: Spectra) -> None:
    """The collection behaves as a sequence and supports lookup by title."""
    assert spectra.titles == ["Spectrum_1", "Spectrum_2"]
    assert spectra["Spectrum_2"] is spectra[1]
    assert isinstance(spectra[1:], Spectra)
    assert spectra[1:].titles == ["Spectrum_2"]
    with pytest.raises(KeyError, match="No item titled"):
        spectra["missing"]


def test_duplicate_titles_are_disambiguated(spectra: Spectra, tmp_path: Path) -> None:
    """Repeated titles get a numeric suffix so that no exported file is overwritten."""
    repeated = Spectra([spectra[0], spectra[0], spectra[0]])
    assert repeated.unique_titles() == ["Spectrum_1", "Spectrum_1_2", "Spectrum_1_3"]
    assert len(repeated.to_csv(tmp_path)) == 3
    assert len(list(tmp_path.glob("*.csv"))) == 3


# --------------------------------------------------------------------------------------------
# Exporting spectra
# --------------------------------------------------------------------------------------------


def test_spectrum_to_df(spectra: Spectra) -> None:
    """A spectrum exports to a DataFrame indexed by the spectral axis."""
    df = spectra[0].to_df("raman_shift")
    assert list(df.columns) == ["Spectrum_1"]
    assert df.index.name == "raman_shift"
    assert len(df) == SPECTRA_LEN


def test_combined_to_df_warns_on_mismatched_axes(
    spectra: Spectra, caplog: pytest.LogCaptureFixture
) -> None:
    """Combining spectra recorded over different axes warns and aligns on their union."""
    with caplog.at_level(logging.WARNING, logger="nanofinderparser.models"):
        df = spectra.to_df()

    assert "do not share a common spectral axis" in caplog.text
    assert list(df.columns) == spectra.titles
    assert len(df) > SPECTRA_LEN  # the union of two different axes


def test_combined_to_df_is_quiet_for_a_shared_axis(
    spectra: Spectra, caplog: pytest.LogCaptureFixture
) -> None:
    """Spectra recorded over the same axis combine cleanly, with no warning."""
    shared = Spectra([spectra[0], spectra[0]])
    with caplog.at_level(logging.WARNING, logger="nanofinderparser.models"):
        df = shared.to_df()

    assert caplog.text == ""
    assert df.shape == (SPECTRA_LEN, 2)
    assert not df.isna().to_numpy().any()


def test_spectra_to_csv_writes_one_file_each(spectra: Spectra, tmp_path: Path) -> None:
    """Exporting separately writes one CSV per spectrum."""
    written = spectra.to_csv(tmp_path, filename="Spectra.mdt", spectral_units="raman_shift")
    assert len(written) == 2
    assert all(path.exists() for path in written)
    assert written[0].name == "Spectra_Spectrum_1.csv"
    assert written[0].read_text().splitlines()[0] == "raman_shift,Spectrum_1"


def test_spectra_to_csv_combined(spectra: Spectra, tmp_path: Path) -> None:
    """Exporting combined writes a single CSV holding every spectrum."""
    written = spectra.to_csv(tmp_path, filename="all", combined=True)
    assert [path.name for path in written] == ["all.csv"]
    header = written[0].read_text().splitlines()[0]
    assert header.split(",")[1:] == spectra.titles


# --------------------------------------------------------------------------------------------
# 2-D maps
# --------------------------------------------------------------------------------------------


def test_image_metadata(maps: Images) -> None:
    """The metadata of a map is read correctly."""
    image = maps["2D_Map"]
    assert isinstance(image, Image)
    assert image.shape == (39, 51)
    assert image.values.shape == image.shape
    assert image.value_unit == "counts"
    assert image.x_axis.units == "micrometer"


def test_image_coordinates_follow_the_axis_calibration(maps: Images) -> None:
    """Coordinates start at the stored offset and advance by the stored step."""
    image = maps[0]
    y_size, x_size = image.shape

    assert image.x_coords.size == x_size
    assert image.y_coords.size == y_size
    assert image.x_coords[0] == pytest.approx(image.x_axis.start)
    assert np.diff(image.x_coords) == pytest.approx(image.x_axis.step)
    assert np.diff(image.y_coords) == pytest.approx(image.y_axis.step)


def test_map_scales_to_whole_counts(maps: Images) -> None:
    """A finely quantized map of counts scales to whole photon counts.

    Counts are integers by nature, so this pins down both the int16 interpretation of the stored
    data and the offset and step of the z axis used to scale it.
    """
    image = maps["2D_Map"]
    assert image.value_unit == "counts"
    assert image.values.min() == pytest.approx(round(image.values.min()), abs=1e-3)


def test_map_values_lie_on_the_stored_int16_lattice() -> None:
    """Every map value is an exact int16 level scaled by the stored z calibration.

    This is the defining property of how maps are stored, and it documents a real limitation of
    the format: a map can only resolve steps as fine as the step of its z axis.
    """
    frame = next(
        f for f in read_mdt_frames(MDT_FOLDER / MIXED_FILE) if isinstance(f, MdtImageFrame)
    )
    levels = (frame.values - frame.z_scale.offset) / frame.z_scale.step

    assert levels == pytest.approx(np.round(levels), abs=1e-6)
    assert levels.min() >= -32768
    assert levels.max() <= 32767


def test_image_orientation_is_x_fast(maps: Images) -> None:
    """Maps are reshaped as (y, x), which yields a spatially smooth image.

    A wrong row/column order scrambles the map, so the mean neighbour difference of the correct
    arrangement is markedly lower than that of the transposed reshape.
    """
    image = maps[0]
    y_size, x_size = image.shape
    assert y_size != x_size, "a square map cannot tell the two orderings apart"

    def roughness(array: np.ndarray) -> float:
        return float(np.abs(np.diff(array, axis=0)).mean() + np.abs(np.diff(array, axis=1)).mean())

    scrambled = image.values.ravel().reshape(x_size, y_size)
    assert roughness(image.values) < roughness(scrambled) / 2


def test_image_to_df_and_csv(maps: Images, tmp_path: Path) -> None:
    """A map exports to a DataFrame and a CSV laid out as the map itself."""
    image = maps[0]
    df = image.to_df()
    assert df.shape == image.shape
    assert df.index.name.startswith("y (")
    assert df.columns.name.startswith("x (")

    written = maps.to_csv(tmp_path)
    assert len(written) == len(maps)
    assert all(path.exists() for path in written)


# --------------------------------------------------------------------------------------------
# Error handling and units
# --------------------------------------------------------------------------------------------


def test_rejects_files_that_are_not_mdt(tmp_path: Path) -> None:
    """A file without the NT-MDT signature is rejected with a clear message."""
    bogus = tmp_path / "bogus.mdt"
    bogus.write_bytes(b"not an mdt file at all, but long enough to pass the length check")
    with pytest.raises(ValueError, match="not an NT-MDT"):
        read_mdt_frames(bogus)


def test_truncated_file_is_rejected(tmp_path: Path) -> None:
    """A file cut short mid-frame is rejected rather than silently returning partial data."""
    truncated = tmp_path / "truncated.mdt"
    truncated.write_bytes((MDT_FOLDER / SPECTRA_FILE).read_bytes()[:5000])
    with pytest.raises(ValueError, match=r"invalid size|truncated"):
        read_mdt_frames(truncated)


@pytest.mark.parametrize(("code", "expected"), [(-1, Units.nm), (-10, Units.raman_shift)])
def test_mdt_spectral_unit_codes(code: int, expected: Units) -> None:
    """The NT-MDT unit codes map to the spectral units used by the library."""
    assert MdtUnit(code).spectral_units is expected


def test_non_spectral_mdt_unit_is_rejected() -> None:
    """A unit that cannot describe a spectral axis raises a clear error."""
    with pytest.raises(ValueError, match="not a valid spectral axis unit"):
        MdtUnit.counts.spectral_units  # noqa: B018
