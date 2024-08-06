"""Handle NanoFinder files."""

# REFERENCE See this project, and how they handle the MDT files:
# https://github.com/symartin/PyMDT/blob/master/MDTfile.py

from pathlib import Path

from nanofinderparser.models import Channel, Mapping
from nanofinderparser.parsers import read_binary_part, read_xml_part

# TODO # ISSUE #1

# TODO Need to handle the unit conversion to "raman_shift" properly (now just cm-1...)


def load_smd_file(file: Path) -> Mapping:
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
    >>> mapping = load_smd_file(smd_file)

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

    return Mapping(**scandata)


# MAPPING_FILE_RAW = Path("_working/Nanofinder_raw_files/mapping_file_hBN.smd")
# MAPPING_FILE_RAW = Path("_working/PL_MoS2WS2_NR.smd")
# MAPPING_FILE_RAW = Path("_working/Raman_BLG.smd")

# mapping_data = load_smd_file(MAPPING_FILE_RAW)
# print(mapping_data.datetime)
# print(mapping_data.step_size)
# print(mapping_data.step_units)
# print(mapping_data.map_size)
# print(mapping_data)
# spectral_axis = mapping_data.get_spectral_axis
# print(spectral_axis)
# print(type(mapping_data.get_spectral_axis))
# # mapping_data._to_spectral_units("cm-1")

# mapping_data.export_to_csv(path=Path("_working/Nanofinder_raw_files"), spectral_units="cm-1")

# mapping_data.export_to_csv(filename="BORRAR.csv", spectral_units="eV")

# print(mapping_data.scanned_frame_parameters.data_calibration.channels[0].channel_info)
