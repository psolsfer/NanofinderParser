import random
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nanofinderparser import load_smd
from nanofinderparser.models import Mapping
from nanofinderparser.parsers import read_xml_part


def reduce_smd_file(
    input_file: Path, output_file: Path, num_signals: int = 4, num_cols: int | None = None
) -> None:
    """
    Reduce an SMD file by keeping only a specified number of random signals and optionally reducing the number of columns.

    Parameters
    ----------
    input_file : Path
        Path to the input SMD file.
    output_file : Path
        Path to save the reduced SMD file.
    num_signals : int, optional
        Number of random signals to keep, by default 4.
    num_cols : int | None, optional
        Number of columns to keep. If None, all columns are kept. By default None.

    Raises
    ------
    ValueError
        If the input file is empty or doesn't contain valid data.
    """
    if input_file.stat().st_size == 0:
        raise ValueError(f"The file {input_file} is empty.")

    # try:
    # Load the SMD file into a Mapping object
    mapping = load_smd(input_file)

    # Get the data as a NumPy array
    data_array = mapping.data

    # Select random rows
    total_signals = data_array.shape[0]
    random_rows = random.sample(range(total_signals), min(num_signals, total_signals))
    reduced_array = data_array[random_rows]

    # Get the spectral axis
    spectral_axis = mapping.get_spectral_axis()

    # Reduce columns if specified
    if num_cols is not None and num_cols < data_array.shape[1]:
        reduced_array = reduced_array[:, :num_cols]
        spectral_axis = spectral_axis[:num_cols]

    # Read the original XML part
    xml_data, _ = read_xml_part(input_file)

    # Modify the XML data to reflect the reduced number of signals and columns
    xml_data["SCANDATA"]["ScannedFrameParameters"]["Stage3DParameters"]["AxisSizeX"] = "2"
    xml_data["SCANDATA"]["ScannedFrameParameters"]["Stage3DParameters"]["AxisSizeY"] = "2"

    if num_cols is not None:
        xml_data["SCANDATA"]["ScannedFrameParameters"]["DataCalibration"]["DataDimentions"][
            "Channel0"
        ]["ChannelSize"] = str(num_cols)
        # Update ChannelAxisArray
        channel_axis_array = " ".join(map(str, spectral_axis))
        xml_data["SCANDATA"]["ScannedFrameParameters"]["DataCalibration"]["DataDimentions"][
            "Channel0"
        ]["ChannelAxisArray"] = channel_axis_array

    # Calculate and update DataBlockSizeBytes
    data_block_size = reduced_array.size * 4  # 4 bytes per float32
    xml_data["SCANDATA"]["ScannedFrameParameters"]["DataBlockSizeBytes"] = str(data_block_size)

    # Write the reduced SMD file
    with output_file.open("wb") as f:
        # Write the modified XML part
        f.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(b"<SCANDATA>\n")
        write_xml_part(f, xml_data["SCANDATA"], indent=2)
        f.write(b"</SCANDATA>\n")

        # Write the binary part
        binary_data = reduced_array.flatten()
        f.write(struct.pack(f"{len(binary_data)}f", *binary_data))

    print(f"Reduced SMD file saved to {output_file}")
    print(f"Original number of signals: {total_signals}")
    print(f"Reduced number of signals: {len(random_rows)}")
    print(f"Number of columns: {reduced_array.shape[1]}")

    # except Exception as e:
    #     raise ValueError(f"Error processing the file {input_file}: {str(e)}") from e


def write_xml_part(file, data: dict[str, Any], indent: int = 0) -> None:
    """
    Write a dictionary as XML to a file.

    Parameters
    ----------
    file : file object
        The file to write to.
    data : dict[str, Any]
        The data to write as XML.
    indent : int, optional
        The current indentation level, by default 0.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            file.write(f"{' ' * indent}<{key}>\n".encode())
            write_xml_part(file, value, indent + 2)
            file.write(f"{' ' * indent}</{key}>\n".encode())
        elif isinstance(value, list):
            for item in value:
                file.write(f"{' ' * indent}<{key}>\n".encode())
                write_xml_part(file, item, indent + 2)
                file.write(f"{' ' * indent}</{key}>\n".encode())
        else:
            file.write(f"{' ' * indent}<{key}>{value}</{key}>\n".encode())


# Example usage
if __name__ == "__main__":
    # input_file = Path("_working/PL_MoS2WS2_NR.smd")
    input_file = Path("_working/Raman_BLG.smd")
    output_file = Path("_working/Raman_BLG_reduced.smd")
    reduce_smd_file(input_file, output_file, num_signals=4, num_cols=10)
