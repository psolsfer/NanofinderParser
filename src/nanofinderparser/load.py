"""Handle NanoFinder files."""

from collections.abc import Generator
from pathlib import Path

from nanofinderparser.models import Channel, Mapping
from nanofinderparser.parsers import read_binary_part, read_xml_part

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


def load_smd_folder(folder_path: Path) -> Generator[Mapping, None, None]:
    """Load SMD files from a folder.

    Parameters
    ----------
    folder_path : Path
        Path to the folder containing SMD files.

    Yields
    ------
    Mapping
        Mapping objects, each representing a loaded SMD file.

    Examples
    --------
    >>> from pathlib import Path
    >>> folder_path = Path("/path/to/smd/files")
    >>> for mapping in load_smd_folder(folder_path):
    ...     process_mapping(mapping)
    """
    folder_path = Path(folder_path)
    smd_files = list(folder_path.glob("*.smd"))

    for file in smd_files:
        yield load_smd(file)
