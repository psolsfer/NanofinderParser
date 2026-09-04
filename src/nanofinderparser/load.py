"""Handle NanoFinder files."""

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal, overload

from numpy.typing import NDArray

from nanofinderparser.models import (
    Channel,
    Image,
    Images,
    KineticSeries,
    Mapping,
    Spectra,
    Spectrum,
)
from nanofinderparser.parsers import (
    S1D_ROOT,
    SMD_ROOT,
    MdtImageFrame,
    MdtSpectrumFrame,
    read_binary_part,
    read_mdt_frames,
    read_xml_part,
)

logger = logging.getLogger(__name__)

# TODO Need to handle the unit conversion to "raman_shift" properly (now just cm-1...)


def _read_frame(file: Path, root: str, other_root: str, other_loader: str) -> dict[str, Any]:
    """Read the XML header and the binary block of an SMD or S1D file.

    Both formats share a layout: a single-rooted XML header listing the scan parameters and the
    detector channels, immediately followed by the raw block of every spectrum. Only the name of
    the root element and the meaning of the spectra differ.

    Parameters
    ----------
    file : Path
        The file to read.
    root : str
        Name of the root element the file is expected to carry.
    other_root : str
        Name of the root element of the sibling format, recognized to give a better error.
    other_loader : str
        Name of the function that reads the sibling format, quoted in that error.

    Returns
    -------
    dict[str, Any]
        The content of the root element, with its channels parsed into
        :class:`~nanofinderparser.models.Channel` objects and the binary block under ``"Data"``,
        ready to build a :class:`~nanofinderparser.models.Mapping` or a
        :class:`~nanofinderparser.models.KineticSeries`.

    Raises
    ------
    KeyError
        If expected keys are missing in the XML data.
    ValueError
        If the root element of the file is not the expected one.
    """
    xml_data, file_position = read_xml_part(file)

    found = next(iter(xml_data), "")
    if found != root:
        hint = f" Use {other_loader}() for those." if found == other_root else ""
        msg = f"{file} has a <{found}> root element, not the <{root}> of this format.{hint}"
        raise ValueError(msg)

    scandata: dict[str, Any] = xml_data[root]

    # Parse channels
    channels_data = scandata["ScannedFrameParameters"]["DataCalibration"].pop("DataDimentions")
    channels = [
        Channel(**value) for key, value in channels_data.items() if key.startswith("Channel")
    ]
    scandata["ScannedFrameParameters"]["DataCalibration"]["Channels"] = channels

    scandata["Data"] = read_binary_part(file, file_position)
    return scandata


def _validate_data_block(
    file: Path, expected: int, found: int, described_as: str, data: NDArray[Any]
) -> NDArray[Any]:
    """Check the binary block of a file against what its XML header declares.

    The header states how many spectra the file holds and how long each one is, but the binary
    block carries no length of its own: it simply runs to the end of the file. Comparing the two
    turns a silent mis-reshape into a clear error.

    Parameters
    ----------
    file : Path
        The file the data was read from, used for the messages.
    expected : int
        Number of values the header describes.
    found : int
        Number of values actually read.
    described_as : str
        How the header describes the expected values, quoted in the error.
    data : NDArray[Any]
        The values read from the file.

    Returns
    -------
    NDArray[Any]
        The data, truncated to `expected` when the file holds more values than declared.

    Raises
    ------
    ValueError
        If the file holds fewer values than its header declares.
    """
    if found < expected:
        msg = (
            f"{file} is truncated: its header describes {expected} values ({described_as}), but "
            f"only {found} could be read."
        )
        raise ValueError(msg)

    if found > expected:
        logger.warning(
            "%s holds %d values more than the %d its header describes; ignoring the extra ones.",
            file,
            found - expected,
            expected,
        )
        return data[:expected]

    return data


def load_smd(file: Path) -> Mapping:
    """Load and parse a Nanofinder SMD file for mappings.

    This is the recommended way to create a Mapping instance.

    Parameters
    ----------
    file : Path
        The path to the SMD file.

    Returns
    -------
    Mapping
        A Mapping object containing the parsed data.

    Raises
    ------
    KeyError
        If expected keys are missing in the XML data.
    ValueError
        If the binary part of the file holds fewer values than its header describes.
    NotImplementedError
        If the file holds more than one detector channel, or more than one acquisition per
        spatial point.
    IOError
        If there's an error reading the file.
    xmltodict.expat.ExpatError
        If there's an error parsing the XML.

    Examples
    --------
    >>> from pathlib import Path
    >>> smd_file = Path("path/to/your/file.smd")
    >>> mapping = load_smd(smd_file)  # doctest: +SKIP

    """
    file = Path(file)

    scandata = _read_frame(file, SMD_ROOT, S1D_ROOT, "load_s1d")

    mapping = Mapping(scandata, source=file)
    _validate_smd_data_block(mapping, file)
    return mapping


def _validate_smd_data_block(mapping: Mapping, file: Path) -> None:
    """Check the binary block against what the XML header of an SMD file declares.

    The header states how many spectra the file holds and how long each one is, but the binary
    block carries no length of its own: it simply runs to the end of the file. Comparing the two
    turns a silent mis-reshape into a clear error.

    Parameters
    ----------
    mapping : Mapping
        The mapping just built from the file. Its data is truncated in place when the file holds
        more values than the header declares.
    file : Path
        The file the mapping was read from, used for the messages.

    Raises
    ------
    NotImplementedError
        If the file holds more than one detector channel, or more than one acquisition per
        spatial point, neither of which is supported yet.
    ValueError
        If the file holds fewer values than its header declares.
    """
    channel = mapping.single_channel()

    x_steps, y_steps, z_steps = mapping.map_steps
    expected = mapping.expected_data_size

    _warn_on_declared_block_size(
        mapping.scanned_frame_parameters.data_block_size_bytes,
        file,
        expected,
        int(mapping.data.itemsize),
    )

    mapping.data = _validate_data_block(
        file,
        expected,
        int(mapping.data.size),
        f"{x_steps} x {y_steps} x {z_steps} points of {channel.channel_size} each",
        mapping.data,
    )


def _warn_on_declared_block_size(
    declared_bytes: int | None, file: Path, expected: int, itemsize: int
) -> None:
    """Warn when the header of a file contradicts itself about the size of its binary block.

    Parameters
    ----------
    declared_bytes : int | None
        Size of the binary block as declared by ``DataBlockSizeBytes``, when present.
    file : Path
        The file the data was read from, used for the message.
    expected : int
        Number of values the scan parameters describe.
    itemsize : int
        Size in bytes of a single stored value.
    """
    if declared_bytes is not None and declared_bytes != expected * itemsize:
        logger.warning(
            "%s declares a data block of %d bytes, but its scan parameters describe %d values "
            "of %d bytes; trusting the scan parameters.",
            file,
            declared_bytes,
            expected,
            itemsize,
        )


@overload
def load_smd_folder(
    folder_path: Path, return_path: Literal[False] = False
) -> Generator[Mapping, None, None]: ...
@overload
def load_smd_folder(
    folder_path: Path, return_path: Literal[True]
) -> Generator[tuple[Mapping, Path], None, None]: ...
def load_smd_folder(
    folder_path: Path,
    return_path: bool = False,
) -> Generator[Mapping | tuple[Mapping, Path], None, None]:
    """Load SMD files from a folder.

    Parameters
    ----------
    folder_path : Path
        Path to the folder containing SMD files.
    return_path : bool, optional
        If True, also yield the file path alongside the loaded mapping,
        as a (mapping, path) tuple. Defaults to False.

    Yields
    ------
    Mapping
        If `return_path` is False (default), yields loaded SMD mappings.
    tuple of Mapping and Path
        If `return_path` is True, yields a tuple of (mapping, file path).

    Examples
    --------
    >>> from pathlib import Path
    >>> folder_path = Path("/path/to/smd/files")
    >>> for mapping in load_smd_folder(folder_path):
    ...     process_mapping(mapping)
    >>> for mapping, path in load_smd_folder(folder_path, return_path=True):
    ...     print(f"{path.name}: {mapping}")
    """
    folder_path = Path(folder_path)
    smd_files = list(folder_path.glob("*.smd"))

    for file in smd_files:
        loaded = load_smd(file)
        yield (loaded, file) if return_path else loaded


def load_s1d(file: Path) -> KineticSeries:
    """Load and parse a NanoFinder S1D file, which holds a kinetic series.

    An S1D file stores a set of spectra measured one after another at a single point of the
    sample, at a fixed period, rather than the spatial scan an SMD file holds. This is the
    recommended way to create a :class:`~nanofinderparser.models.KineticSeries` instance.

    Parameters
    ----------
    file : Path
        The path to the S1D file.

    Returns
    -------
    KineticSeries
        A KineticSeries object containing the parsed data.

    Raises
    ------
    KeyError
        If expected keys are missing in the XML data.
    ValueError
        If the file is not an S1D file, or if its binary part holds fewer values than its header
        describes.
    NotImplementedError
        If the file holds more than one detector channel, or spans more than one stage point.
    IOError
        If there's an error reading the file.
    xmltodict.expat.ExpatError
        If there's an error parsing the XML.

    Examples
    --------
    >>> from pathlib import Path
    >>> s1d_file = Path("path/to/your/file.s1d")
    >>> series = load_s1d(s1d_file)  # doctest: +SKIP
    >>> series.series_size, series.kinetic_cycle_time  # doctest: +SKIP
    (500, 0.2304)

    """
    file = Path(file)

    spectradata = _read_frame(file, S1D_ROOT, SMD_ROOT, "load_smd")

    series = KineticSeries(spectradata, source=file)
    _validate_s1d_data_block(series, file)
    return series


def _validate_s1d_data_block(series: KineticSeries, file: Path) -> None:
    """Check the binary block against what the XML header of an S1D file declares.

    Parameters
    ----------
    series : KineticSeries
        The series just built from the file. Its data is truncated in place when the file holds
        more values than the header declares.
    file : Path
        The file the series was read from, used for the messages.

    Raises
    ------
    NotImplementedError
        If the file holds more than one detector channel, or spans more than one stage point,
        neither of which is supported.
    ValueError
        If the file holds fewer values than its header declares.
    """
    channel = series.single_channel()
    expected = series.expected_data_size

    _warn_on_declared_block_size(
        series.scanned_frame_parameters.data_block_size_bytes,
        file,
        expected,
        int(series.data.itemsize),
    )

    series.data = _validate_data_block(
        file,
        expected,
        int(series.data.size),
        f"{channel.series_size} spectra of {channel.channel_size} points each",
        series.data,
    )


@overload
def load_s1d_folder(
    folder_path: Path, return_path: Literal[False] = False
) -> Generator[KineticSeries, None, None]: ...
@overload
def load_s1d_folder(
    folder_path: Path, return_path: Literal[True]
) -> Generator[tuple[KineticSeries, Path], None, None]: ...
def load_s1d_folder(
    folder_path: Path,
    return_path: bool = False,
) -> Generator[KineticSeries | tuple[KineticSeries, Path], None, None]:
    """Load S1D files from a folder.

    Parameters
    ----------
    folder_path : Path
        Path to the folder containing S1D files.
    return_path : bool, optional
        If True, also yield the file path alongside the loaded series,
        as a (series, path) tuple. Defaults to False.

    Yields
    ------
    KineticSeries
        If `return_path` is False (default), yields the loaded kinetic series.
    tuple of KineticSeries and Path
        If `return_path` is True, yields a tuple of (series, file path).

    Examples
    --------
    >>> from pathlib import Path
    >>> folder_path = Path("/path/to/s1d/files")
    >>> for series in load_s1d_folder(folder_path):
    ...     process_series(series)
    >>> for series, path in load_s1d_folder(folder_path, return_path=True):
    ...     print(f"{path.name}: {len(series)} spectra")
    """
    folder_path = Path(folder_path)

    for file in folder_path.glob("*.s1d"):
        loaded = load_s1d(file)
        yield (loaded, file) if return_path else loaded


def _read_mdt_frames_by_kind(
    file: Path,
) -> tuple[list[MdtSpectrumFrame], list[MdtImageFrame]]:
    """Read an MDT file once and split its frames by kind.

    Parameters
    ----------
    file : Path
        The path to the MDT file.

    Returns
    -------
    tuple[list[MdtSpectrumFrame], list[MdtImageFrame]]
        The spectrum frames and the map frames, each in file order.
    """
    frames = read_mdt_frames(file)
    return (
        [frame for frame in frames if isinstance(frame, MdtSpectrumFrame)],
        [frame for frame in frames if isinstance(frame, MdtImageFrame)],
    )


def load_mdt(file: Path) -> Spectra:
    """Load and parse a NanoFinder MDT file of individual spectra.

    Unlike SMD files, which hold a spectrum per point of a spatial scan, MDT files hold a set of
    standalone spectra with no stage coordinates. This is the recommended way to create a
    :class:`~nanofinderparser.models.Spectra` collection.

    Parameters
    ----------
    file : Path
        The path to the MDT file.

    Returns
    -------
    Spectra
        A collection with one :class:`~nanofinderparser.models.Spectrum` per spectrum stored in
        the file, in file order.

    Raises
    ------
    ValueError
        If the file is not an NT-MDT file, or if one of its frames is inconsistent.
    NotImplementedError
        If a frame stores a layout that is not supported yet.
    OSError
        If there's an error reading the file.

    Notes
    -----
    MDT files may also hold 2-D scalar maps, which this function ignores. Use
    :func:`load_mdt_images` to read those.

    Examples
    --------
    >>> from pathlib import Path
    >>> mdt_file = Path("path/to/your/file.mdt")
    >>> spectra = load_mdt(mdt_file)  # doctest: +SKIP
    >>> spectra.titles  # doctest: +SKIP
    ['Spectrum_1', 'Spectrum_2']

    """
    file = Path(file)
    spectrum_frames, image_frames = _read_mdt_frames_by_kind(file)

    if image_frames:
        logger.info(
            "%s holds %d map frame(s) alongside %d spectra; use load_mdt_images() to read them.",
            file,
            len(image_frames),
            len(spectrum_frames),
        )

    return Spectra([Spectrum.from_mdt_frame(frame) for frame in spectrum_frames], source=file)


def load_mdt_images(file: Path) -> Images:
    """Load the 2-D scalar maps stored in a NanoFinder MDT file.

    These are the maps NanoFinder writes either from a direct measurement (for instance a PL
    intensity map) or from fitting each spectrum of a mapping, such as the position, intensity,
    or FWHM of a peak. This is the recommended way to create an
    :class:`~nanofinderparser.models.Images` collection.

    Parameters
    ----------
    file : Path
        The path to the MDT file.

    Returns
    -------
    Images
        A collection with one :class:`~nanofinderparser.models.Image` per map stored in the file,
        in file order.

    Raises
    ------
    ValueError
        If the file is not an NT-MDT file, or if one of its frames is inconsistent.
    OSError
        If there's an error reading the file.

    Notes
    -----
    Individual spectra held by the same file are ignored; use :func:`load_mdt` to read those.

    Examples
    --------
    >>> from pathlib import Path
    >>> images = load_mdt_images(Path("path/to/your/file.mdt"))  # doctest: +SKIP
    >>> images.titles  # doctest: +SKIP
    ['G Peak intensity (Lorentz)', 'G Peak position (Lorentz)']

    """
    file = Path(file)
    spectrum_frames, image_frames = _read_mdt_frames_by_kind(file)

    if spectrum_frames:
        logger.info(
            "%s holds %d spectrum frame(s) alongside %d maps; use load_mdt() to read them.",
            file,
            len(spectrum_frames),
            len(image_frames),
        )

    return Images([Image.from_mdt_frame(frame) for frame in image_frames], source=file)


def load_mdt_file(file: Path) -> tuple[Spectra, Images]:
    """Load the spectra and the 2-D maps of a NanoFinder MDT file in a single pass.

    :func:`load_mdt` and :func:`load_mdt_images` each read and decode the whole file, so asking
    for both means parsing it twice. Use this function when you want everything a file holds.

    Parameters
    ----------
    file : Path
        The path to the MDT file.

    Returns
    -------
    tuple[Spectra, Images]
        The individual spectra and the 2-D maps stored in the file, each in file order. Either
        collection may be empty.

    Raises
    ------
    ValueError
        If the file is not an NT-MDT file, or if one of its frames is inconsistent.
    NotImplementedError
        If a spectrum frame stores a layout that is not supported yet.
    OSError
        If there's an error reading the file.

    Examples
    --------
    >>> from pathlib import Path
    >>> spectra, images = load_mdt_file(Path("path/to/your/file.mdt"))  # doctest: +SKIP
    >>> len(spectra), len(images)  # doctest: +SKIP
    (2, 1)

    """
    file = Path(file)
    spectrum_frames, image_frames = _read_mdt_frames_by_kind(file)

    return (
        Spectra([Spectrum.from_mdt_frame(frame) for frame in spectrum_frames], source=file),
        Images([Image.from_mdt_frame(frame) for frame in image_frames], source=file),
    )


@overload
def load_mdt_folder(
    folder_path: Path, return_path: Literal[False] = False
) -> Generator[Spectra, None, None]: ...
@overload
def load_mdt_folder(
    folder_path: Path, return_path: Literal[True]
) -> Generator[tuple[Spectra, Path], None, None]: ...
def load_mdt_folder(
    folder_path: Path,
    return_path: bool = False,
) -> Generator[Spectra | tuple[Spectra, Path], None, None]:
    """Load MDT files from a folder.

    Parameters
    ----------
    folder_path : Path
        Path to the folder containing MDT files.
    return_path : bool, optional
        If True, also yield the file path alongside the loaded spectra,
        as a (spectra, path) tuple. Defaults to False.

    Yields
    ------
    Spectra
        If `return_path` is False (default), yields the loaded collections of spectra.
    tuple of Spectra and Path
        If `return_path` is True, yields a tuple of (spectra, file path).

    Examples
    --------
    >>> from pathlib import Path
    >>> folder_path = Path("/path/to/mdt/files")
    >>> for spectra in load_mdt_folder(folder_path):
    ...     process_spectra(spectra)
    >>> for spectra, path in load_mdt_folder(folder_path, return_path=True):
    ...     print(f"{path.name}: {spectra.titles}")
    """
    folder_path = Path(folder_path)
    mdt_files = list(folder_path.glob("*.mdt"))

    for file in mdt_files:
        loaded = load_mdt(file)
        yield (loaded, file) if return_path else loaded
