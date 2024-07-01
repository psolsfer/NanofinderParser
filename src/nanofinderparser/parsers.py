"""Parse the different parts of Nanofinder files."""

import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, overload

import xmltodict


def read_xml_part(file: Path, position: int = 0) -> tuple[dict[str, Any], int]:
    """Read the XML part of a file.

    Parameters
    ----------
    file : Path
        The file to read.
    position : int, optional
        The position in the file where the binary part starts, by default 0.

    Returns
    -------
    xml_data : dict[str, Any]
        The read data as a dictionary.
    position : int
        The current position in the file.
    """
    with Path.open(file, "rb") as f:
        xml_content = b""
        first_tag = None
        f.seek(position)  # Move to the indicated position
        for line in f:
            if first_tag is None and not line.strip().startswith(b"<?xml"):
                first_tag = line.split()[0][1:-1]

            xml_content += line

            if first_tag and line.strip().startswith(b"</" + first_tag + b">"):
                break

        xml_data = xmltodict.parse(xml_content)

        return xml_data, f.tell()


@overload
def read_binary_part(
    file: Path, position: int, data_format: Literal["c", "s", "p"]
) -> Sequence[bytes]: ...


@overload
def read_binary_part(
    file: Path,
    position: int,
    data_format: Literal["b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"],
) -> Sequence[int]: ...


@overload
def read_binary_part(
    file: Path, position: int, data_format: Literal["f", "d"]
) -> Sequence[float]: ...


@overload
def read_binary_part(
    file: Path, position: int = 0, data_format: str = "f"
) -> Sequence[float | int | bytes]: ...


def read_binary_part(
    file: Path, position: int = 0, data_format: str = "f"
) -> Sequence[float | int | bytes]:
    """Read the binary part of a file.

    Parameters
    ----------
    file : Path
        The file to read.
    position : int, optional
        The position in the file where the binary part starts, by default 0.
    data_format : str | bytes, optional
        The format of the data, by default "f". "f" if data is composed of floats, "i" if it's
        composed of integers. See https://docs.python.org/3/library/struct.html#struct-alignment for
        further information.
        # TODO 'data_format' can be more complex, like ">bhl"...

    Returns
    -------
    data : list[float | int | bytes]
        The read data as a list. The type depends on the 'data_format'.
    """
    data_size = struct.calcsize(data_format)

    data: list[float] = []
    with Path.open(file, "rb") as f:
        f.seek(position)  # Move to the indicated position
        data_bin = f.read()

    length_of_binary_part = len(data_bin)
    data = []
    for i in range(0, length_of_binary_part, data_size):
        chunk = data_bin[i : i + data_size]
        if len(chunk) < data_size:
            break
        data.extend(struct.unpack(data_format, chunk))
    return data
