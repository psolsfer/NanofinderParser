"""Handle NanoFinder files."""

# REFERENCE See this project, and how they handle the MDT files:
# https://github.com/symartin/PyMDT/blob/master/MDTfile.py

from pathlib import Path

from nanofinderparser.models import Channel, Mapping
from nanofinderparser.parsers import read_binary_part, read_xml_part

# TODO # ISSUE #1

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
    >>> mapping = load_smd(smd_file)

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


def _load_smd_folder_sequential(folder_path: Path) -> list[Mapping]:
    """Load SMD files from a folder sequentially.

    Parameters
    ----------
    folder_path : Path
        Path to the folder containing SMD files.

    Returns
    -------
    list[Mapping]
        List of Mapping objects, each representing a loaded SMD file.
    """
    smd_files = list(folder_path.glob("*.smd"))
    return [load_smd(file) for file in smd_files]


def _load_smd_folder_parallel(folder_path: Path) -> list[Mapping]:
    """Load SMD files from a folder in parallel.

    Parameters
    ----------
    folder_path : Path
        Path to the folder containing SMD files.

    Returns
    -------
    list[Mapping]
        List of Mapping objects, each representing a loaded SMD file.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    smd_files = list(folder_path.glob("*.smd"))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        mapping_data = list(executor.map(load_smd, smd_files))
    return mapping_data


def load_smd_folder(folder_path: Path, parallel_threshold: int | None = 10) -> list[Mapping]:
    """Load SMD files from a folder, automatically choosing between sequential and parallel loading.

    This function will use parallel loading if the number of SMD files in the folder is greater than
    or equal to the parallel_threshold. Otherwise, it will use sequential loading.

    Parameters
    ----------
    folder_path : Path
        Path to the folder containing SMD files.
    parallel_threshold : int | None, optional
        The minimum number of files required to trigger parallel loading, by default 10.
        - Parallel loading is used if the number of files is greater than or equal to the threshold.
        - Sequential loading can be forced with 'parallel_threshold=None'.
        - Parallel loading can be forced with 'parallel_threshold=0'.


    Returns
    -------
    list[Mapping]
        List of Mapping objects, each representing a loaded SMD file.

    Notes
    -----
    - This function handles multiprocessing setup automatically. It should be called from a script
      that is run as the main program for multiprocessing to work correctly across all platforms.
    - Parallel loading generally performs better when there are many files and/or the files are
    large. However, for a small number of files or very small files, the overhead of setting up
    parallel processing might outweigh the benefits.
    - The optimal threshold for parallel loading can vary depending on the system and file
    characteristics. Experimenting with different threshold values may help optimize performance for
    specific use cases.

    Examples
    --------
    >>> from pathlib import Path
    >>> folder_path = Path("/path/to/smd/files")
    >>> # Adaptive loading (uses parallel if 10 or more files)
    >>> mappings = load_smd_folder(folder_path)
    >>> # Force sequential loading
    >>> mappings_seq = load_smd_folder(folder_path, parallel_threshold=None)
    >>> # Force parallel loading
    >>> mappings_par = load_smd_folder(folder_path, parallel_threshold=0)
    >>> # Custom threshold for parallel loading
    >>> mappings_custom = load_smd_folder(folder_path, parallel_threshold=20)
    """
    if parallel_threshold is None:
        return _load_smd_folder_sequential(folder_path)

    smd_files = list(folder_path.glob("*.smd"))
    if len(smd_files) >= parallel_threshold:
        import multiprocessing

        if multiprocessing.current_process().name == "MainProcess":
            multiprocessing.freeze_support()
        return _load_smd_folder_parallel(folder_path)
    else:
        return _load_smd_folder_sequential(folder_path)


# MAPPING_FILE_RAW = Path("_working/Nanofinder_raw_files/mapping_file_hBN.smd")
# MAPPING_FILE_RAW = Path("_working/PL_MoS2WS2_NR.smd")
# MAPPING_FILE_RAW = Path("_working/Raman_BLG.smd")

# mapping_data = load_smd(MAPPING_FILE_RAW)
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
