"""Handle NanoFinder files."""

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Literal, overload

from nanofinderparser.models import Channel, Image, Images, Mapping, Spectra, Spectrum
from nanofinderparser.parsers import (
    MdtImageFrame,
    MdtSpectrumFrame,
    read_binary_part,
    read_mdt_frames,
    read_xml_part,
)

logger = logging.getLogger(__name__)

# TODO Need to handle the unit conversion to "raman_shift" properly (now just cm-1...)


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

    # 1st part of the mapping file is xml
    xml_data, file_position = read_xml_part(file)
    scandata = xml_data["SCANDATA"]

    # Parse channels
    channels_data = scandata["ScannedFrameParameters"]["DataCalibration"].pop("DataDimentions")
    channels = []
    for key, value in channels_data.items():
        if key.startswith("Channel"):
            channels.append(Channel(**value))
    scandata["ScannedFrameParameters"]["DataCalibration"]["Channels"] = channels

    # 2nd part of the mapping file is binary
    binary_data = read_binary_part(file, file_position)
    scandata["Data"] = binary_data

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
    channels = mapping.scanned_frame_parameters.data_calibration.channels
    if len(channels) != 1:
        msg = (
            f"{file} declares {len(channels)} detector channels; only single-channel SMD files "
            "are supported."
        )
        raise NotImplementedError(msg)

    channel = channels[0]
    if channel.series_size != 1:
        msg = (
            f"{file} stores {channel.series_size} acquisitions per spatial point (SeriesSize); "
            "only one per point is supported."
        )
        raise NotImplementedError(msg)

    x_steps, y_steps, z_steps = mapping.map_steps
    expected = x_steps * y_steps * z_steps * channel.series_size * channel.channel_size
    found = int(mapping.data.size)

    declared_bytes = mapping.scanned_frame_parameters.data_block_size_bytes
    if declared_bytes is not None and declared_bytes != expected * mapping.data.itemsize:
        logger.warning(
            "%s declares a data block of %d bytes, but its scan parameters describe %d values "
            "of %d bytes; trusting the scan parameters.",
            file,
            declared_bytes,
            expected,
            mapping.data.itemsize,
        )

    if found < expected:
        msg = (
            f"{file} is truncated: its header describes {expected} values "
            f"({x_steps} x {y_steps} x {z_steps} points of {channel.channel_size} each), but "
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
        mapping.data = mapping.data[:expected]


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
