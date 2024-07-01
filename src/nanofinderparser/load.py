"""Handle NanoFinder files."""

# REFERENCE See this project, and how they handle the MDT files:
# https://github.com/symartin/PyMDT/blob/master/MDTfile.py

from pathlib import Path

from nanofinderparser.models import Mapping
from nanofinderparser.parsers import read_binary_part, read_xml_part

# MAPPING_FILE_RAW = Path("_working/Nanofinder_raw_files/mapping_file_hBN.smd")

# TODO # ISSUE #45

# TODO Need to handle the unit conversion to "raman_shift" properly (now just cm-1...)


def load_smd_file(file: Path) -> Mapping:  # NOTE Will return a Mapping Model.
    """Parse a Nanofinder smd file for mappings.

    Parameters
    ----------
    file : Path
        The path to the smd file.

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
    """
    # 1st part of the mapping file is xml
    xml_data, file_position = read_xml_part(file)

    # NOTE Assuming there's only one channel called "Channel0"
    channel0_data = xml_data["SCANDATA"]["ScannedFrameParameters"]["DataCalibration"][
        "DataDimentions"
    ].pop("Channel0")
    xml_data["SCANDATA"]["ScannedFrameParameters"]["Channel"] = channel0_data

    # 2nd part of the mapping file is binary
    binary_data = read_binary_part(file, file_position)

    xml_data["SCANDATA"]["Data"] = binary_data
    return Mapping(**xml_data["SCANDATA"])


# mapping_data = load_smd_file(MAPPING_FILE_RAW)
# # mapping_data._to_spectral_units("cm-1")

# mapping_data.export_to_csv(path=Path("_working/Nanofinder_raw_files"), spectral_units="cm-1")
