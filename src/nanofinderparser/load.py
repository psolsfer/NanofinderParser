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
    IOError
        If there's an error reading the file.
    xmltodict.expat.ExpatError
        If there's an error parsing the XML.

    Examples
    --------
    >>> from pathlib import Path
    >>> smd_file = Path("path/to/your/file.smd")
    >>> mapping = load_smd(smd_file) # doctest: +SKIP

    """
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

    return Mapping(scandata)


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
    frames = read_mdt_frames(file)
    spectrum_frames = [frame for frame in frames if isinstance(frame, MdtSpectrumFrame)]

    skipped = len(frames) - len(spectrum_frames)
    if skipped:
        logger.info(
            "%s holds %d map frame(s) alongside %d spectra; use load_mdt_images() to read them.",
            file,
            skipped,
            len(spectrum_frames),
        )

    return Spectra([Spectrum.from_mdt_frame(frame) for frame in spectrum_frames], source=Path(file))


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
    frames = read_mdt_frames(file)
    image_frames = [frame for frame in frames if isinstance(frame, MdtImageFrame)]

    skipped = len(frames) - len(image_frames)
    if skipped:
        logger.info(
            "%s holds %d spectrum frame(s) alongside %d maps; use load_mdt() to read them.",
            file,
            skipped,
            len(image_frames),
        )

    return Images([Image.from_mdt_frame(frame) for frame in image_frames], source=Path(file))


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
