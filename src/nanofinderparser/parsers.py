"""Parse the different parts of Nanofinder files."""

import logging
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, overload
from xml.parsers.expat import ExpatError

import numpy as np
import xmltodict
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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


# ----------------------------------------------------------------------------------------------
# NT-MDT ".mdt" files (individual spectra rather than mappings)
# ----------------------------------------------------------------------------------------------

# Signature found at the start of every NT-MDT ``.mdt`` file.
MDT_MAGIC: Final[bytes] = b"\x01\xb0\x93\xff"

# Bytes taken by the file header, before the first frame.
_MDT_FILE_HEADER_SIZE: Final[int] = 33

# Offset of the ``uint16`` holding the index of the last frame.
_MDT_FRAME_COUNT_OFFSET: Final[int] = 12

# Bytes taken by the fixed part of a frame header.
_MDT_FRAME_HEADER_SIZE: Final[int] = 22

# Bytes taken by a single axis calibration record (``float32``, ``float32``, ``int16``).
_MDT_AXIS_SCALE_SIZE: Final[int] = 10

# Offset, inside the frame body, of the point/array counts.
_MDT_RESOLUTION_OFFSET: Final[int] = 32

# Bytes between the end of the frame variables and the start of the data arrays.
_MDT_DATA_HEADER_SIZE: Final[int] = 8

# Frame type written by NanoFinder for individual spectra.
_MDT_SPECTRUM_FRAME_TYPE: Final[int] = 2

# Frame type written by NanoFinder for 2-D scalar maps (intensity or fitted-parameter maps).
_MDT_IMAGE_FRAME_TYPE: Final[int] = 0

# Bytes per stored value in a spectrum frame (``float32``).
_MDT_SPECTRUM_ITEM_SIZE: Final[int] = 4

# Bytes per stored value in an image frame (signed ``int16``, scaled by the z axis).
_MDT_IMAGE_ITEM_SIZE: Final[int] = 2


@dataclass(frozen=True, slots=True)
class MdtAxisScale:
    """Linear calibration of a single axis of an MDT frame.

    Attributes
    ----------
    offset : float
        Value of the axis at the first point.
    step : float
        Increment of the axis between consecutive points.
    unit_code : int
        Raw NT-MDT unit code. See :class:`~nanofinderparser.units.MdtUnit`.
    """

    offset: float
    step: float
    unit_code: int


@dataclass(frozen=True, slots=True)
class MdtFrameBase:
    """Fields shared by every frame of an NT-MDT ``.mdt`` file.

    Attributes
    ----------
    index : int
        Zero-based position of the frame within the file.
    frame_type : int
        Raw NT-MDT frame type code. NanoFinder writes ``2`` for individual spectra and ``0`` for
        2-D scalar maps.
    version : tuple[int, int]
        ``(major, minor)`` version of the frame layout.
    measured_at : datetime | None
        Acquisition timestamp, or None when the frame stores an invalid date.
    x_scale, y_scale, z_scale : MdtAxisScale
        Calibration of the three axes of the frame.
    title : str
        Name given to the frame in NanoFinder.
    comment : str
        Raw XML ``FrameComment`` block, as stored in the file.
    text_comment : str
        Free-text comment extracted from ``comment``.
    laser_wavelength_nm : float | None
        Excitation wavelength in nm, extracted from ``comment``, or None when absent.
    """

    index: int
    frame_type: int
    version: tuple[int, int]
    measured_at: datetime | None
    x_scale: MdtAxisScale
    y_scale: MdtAxisScale
    z_scale: MdtAxisScale
    title: str
    comment: str
    text_comment: str
    laser_wavelength_nm: float | None


@dataclass(frozen=True, slots=True)
class MdtSpectrumFrame(MdtFrameBase):
    """A frame holding an individual spectrum (NT-MDT frame type 2).

    Attributes
    ----------
    point_count : int
        Number of points of each stored array.
    array_count : int
        Number of stored arrays. NanoFinder writes ``2``: the spectral axis and the intensities.
    arrays : NDArray[np.float64]
        Stored data, of shape ``(array_count, point_count)``. Row 0 is the spectral axis, in the
        units given by ``x_scale``; row 1 the intensities, in the units given by ``z_scale``.

    Notes
    -----
    The values are stored as ``float32`` and are already in physical units, so the ``offset`` and
    ``step`` of the axis calibrations are not applied.
    """

    point_count: int
    array_count: int
    arrays: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MdtImageFrame(MdtFrameBase):
    """A frame holding a 2-D scalar map (NT-MDT frame type 0).

    NanoFinder uses these frames both for maps measured directly (for instance a PL intensity
    map) and for maps of parameters obtained by fitting each spectrum of a mapping, such as the
    position or the FWHM of a peak.

    Attributes
    ----------
    x_size : int
        Number of points along the x (fast) axis.
    y_size : int
        Number of points along the y (slow) axis.
    values : NDArray[np.float64]
        The map, of shape ``(y_size, x_size)``, in the units given by ``z_scale``.

    Notes
    -----
    The values are stored as signed ``int16`` and are converted to physical units on reading, as
    ``z_scale.offset + raw * z_scale.step``. The x axis is the fast one, so the raw data is read
    row by row.
    """

    x_size: int
    y_size: int
    values: NDArray[np.float64]


# Any kind of frame that can be decoded from a ``.mdt`` file.
type MdtAnyFrame = MdtSpectrumFrame | MdtImageFrame


def _read_mdt_axis_scale(body: bytes, offset: int) -> MdtAxisScale:
    """Read one axis calibration record from a frame body.

    Parameters
    ----------
    body : bytes
        The frame body (everything after the 22-byte frame header).
    offset : int
        Offset, inside `body`, at which the record starts.

    Returns
    -------
    MdtAxisScale
        The decoded calibration.
    """
    axis_offset, step, unit_code = struct.unpack_from("<ffh", body, offset)
    return MdtAxisScale(offset=axis_offset, step=step, unit_code=unit_code)


def _read_mdt_string(body: bytes, offset: int, encoding: str) -> tuple[str, int]:
    """Read a ``uint32``-length-prefixed string from a frame body.

    Parameters
    ----------
    body : bytes
        The frame body.
    offset : int
        Offset, inside `body`, of the 4-byte length prefix.
    encoding : str
        Codec used to decode the bytes (``"cp1252"`` for titles, ``"utf-16-le"`` for comments).

    Returns
    -------
    text : str
        The decoded string. Empty when the frame is truncated at this point.
    position : int
        Offset just past the string, to continue reading from.
    """
    if offset + 4 > len(body):
        return "", offset

    (length,) = struct.unpack_from("<I", body, offset)
    start = offset + 4
    end = start + length
    if end > len(body):
        return "", len(body)

    return body[start:end].decode(encoding, errors="replace").rstrip("\x00"), end


def _parse_mdt_comment(comment: str) -> tuple[str, float | None]:
    """Extract the free text and the laser wavelength from a frame comment.

    Parameters
    ----------
    comment : str
        Raw XML ``FrameComment`` block.

    Returns
    -------
    text_comment : str
        The free-text comment, empty when absent.
    laser_wavelength_nm : float | None
        The excitation wavelength in nm, or None when absent or unparsable.
    """
    if not comment.strip():
        return "", None

    try:
        parsed = xmltodict.parse(comment)
    except ExpatError:
        logger.warning("Could not parse the XML frame comment of an MDT frame.")
        return "", None

    frame_comment = parsed.get("FrameComment") or {}
    text_comment = frame_comment.get("TextComment") or ""
    parameters = frame_comment.get("Parameters") or {}
    raw_wavelength = parameters.get("SWLaserWL")

    if raw_wavelength is None:
        return text_comment, None

    try:
        return text_comment, float(raw_wavelength)
    except (TypeError, ValueError):
        logger.warning("Ignoring unparsable laser wavelength %r in an MDT frame.", raw_wavelength)
        return text_comment, None


def _read_mdt_value_counts(body: bytes, var_size: int, index: int) -> tuple[int, int]:
    """Determine how many values a frame stores, and how they are grouped.

    A frame records these counts twice: once among the frame variables, and once in the short
    header that immediately precedes the data. NanoFinder does not always fill in the first
    copy, leaving it at zero, so the copy sitting next to the data is the one to trust.

    Parameters
    ----------
    body : bytes
        The frame body (everything after the 22-byte frame header).
    var_size : int
        Size of the frame variables, which is where the data header starts.
    index : int
        Zero-based position of the frame within the file, used for error messages.

    Returns
    -------
    tuple[int, int]
        For a spectrum frame, the number of points and the number of stored arrays. For an image
        frame, the size of the x and y axes.

    Raises
    ------
    ValueError
        If both copies are filled in and disagree, or if neither gives a usable count.
    """
    from_data = struct.unpack_from("<HH", body, var_size + 2)
    from_vars = struct.unpack_from("<HH", body, _MDT_RESOLUTION_OFFSET)

    if 0 in from_data:
        if 0 in from_vars:
            msg = f"MDT frame {index} does not declare how many values it stores."
            raise ValueError(msg)
        return from_vars

    if 0 not in from_vars and from_vars != from_data:
        msg = (
            f"Inconsistent MDT frame {index}: the frame variables declare "
            f"{from_vars[0]} x {from_vars[1]} values, but the data header declares "
            f"{from_data[0]} x {from_data[1]}."
        )
        raise ValueError(msg)

    return from_data


def _read_mdt_datetime(fields: tuple[int, int, int, int, int, int]) -> datetime | None:
    """Build the acquisition timestamp from the raw frame header fields.

    Parameters
    ----------
    fields : tuple[int, int, int, int, int, int]
        The ``(year, month, day, hour, minute, second)`` values read from the frame header.

    Returns
    -------
    datetime | None
        The timestamp, or None when the stored values are not a valid date.
    """
    try:
        return datetime(*fields)  # noqa: DTZ001 - the file stores no timezone
    except ValueError:
        logger.warning("Ignoring invalid date %r in an MDT frame.", fields)
        return None


def _read_mdt_frame(buffer: bytes, offset: int, index: int) -> tuple[MdtAnyFrame | None, int]:
    """Read a single frame starting at `offset`.

    Parameters
    ----------
    buffer : bytes
        The whole content of the ``.mdt`` file.
    offset : int
        Offset at which the frame starts.
    index : int
        Zero-based position of the frame within the file.

    Returns
    -------
    frame : MdtSpectrumFrame | MdtImageFrame | None
        The decoded frame, or None when its type is not supported.
    next_offset : int
        Offset at which the following frame starts.

    Raises
    ------
    ValueError
        If the frame is truncated or its internal size fields are inconsistent.
    """
    if offset + _MDT_FRAME_HEADER_SIZE > len(buffer):
        msg = f"MDT frame {index} starts past the end of the file."
        raise ValueError(msg)

    (
        frame_size,
        frame_type,
        major,
        minor,
        year,
        month,
        day,
        hour,
        minute,
        second,
        var_size,
    ) = struct.unpack_from("<IHBBHHHHHHH", buffer, offset)

    if frame_size < _MDT_FRAME_HEADER_SIZE or offset + frame_size > len(buffer):
        msg = f"MDT frame {index} declares an invalid size of {frame_size} bytes."
        raise ValueError(msg)

    next_offset = offset + frame_size
    body = buffer[offset + _MDT_FRAME_HEADER_SIZE : next_offset]

    if frame_type not in (_MDT_SPECTRUM_FRAME_TYPE, _MDT_IMAGE_FRAME_TYPE):
        logger.warning("Skipping MDT frame %d: unsupported frame type %d.", index, frame_type)
        return None, next_offset

    x_scale = _read_mdt_axis_scale(body, 0)
    y_scale = _read_mdt_axis_scale(body, _MDT_AXIS_SCALE_SIZE)
    z_scale = _read_mdt_axis_scale(body, 2 * _MDT_AXIS_SCALE_SIZE)

    if var_size + _MDT_DATA_HEADER_SIZE > len(body):
        msg = f"MDT frame {index} is too short to hold its data header."
        raise ValueError(msg)

    x_size, y_size = _read_mdt_value_counts(body, var_size, index)

    is_spectrum = frame_type == _MDT_SPECTRUM_FRAME_TYPE
    item_size = _MDT_SPECTRUM_ITEM_SIZE if is_spectrum else _MDT_IMAGE_ITEM_SIZE
    data_start = var_size + _MDT_DATA_HEADER_SIZE
    value_count = x_size * y_size
    if data_start + item_size * value_count > len(body):
        msg = f"MDT frame {index} is truncated: its data block does not fit in the frame."
        raise ValueError(msg)

    raw = np.frombuffer(
        body, dtype="<f4" if is_spectrum else "<i2", count=value_count, offset=data_start
    ).astype(np.float64)

    title, position = _read_mdt_string(body, data_start + item_size * value_count, "cp1252")
    comment, _ = _read_mdt_string(body, position, "utf-16-le")
    text_comment, laser_wavelength_nm = _parse_mdt_comment(comment)

    shared: dict[str, Any] = {
        "index": index,
        "frame_type": frame_type,
        "version": (major, minor),
        "measured_at": _read_mdt_datetime((year, month, day, hour, minute, second)),
        "x_scale": x_scale,
        "y_scale": y_scale,
        "z_scale": z_scale,
        "title": title,
        "comment": comment,
        "text_comment": text_comment,
        "laser_wavelength_nm": laser_wavelength_nm,
    }

    frame: MdtAnyFrame
    if is_spectrum:
        # Spectrum values are stored directly in physical units.
        frame = MdtSpectrumFrame(
            **shared,
            point_count=x_size,
            array_count=y_size,
            arrays=raw.reshape(y_size, x_size),
        )
    else:
        # Image values are stored as int16 and scaled through the z axis calibration.
        frame = MdtImageFrame(
            **shared,
            x_size=x_size,
            y_size=y_size,
            values=(z_scale.offset + raw * z_scale.step).reshape(y_size, x_size),
        )
    return frame, next_offset


def read_mdt_frames(file: Path) -> list[MdtAnyFrame]:
    """Read every supported frame of an NT-MDT ``.mdt`` file.

    Parameters
    ----------
    file : Path
        The file to read.

    Returns
    -------
    list[MdtSpectrumFrame | MdtImageFrame]
        The decoded frames, in file order. A single file may hold both kinds. Frames of an
        unsupported type are skipped with a warning.

    Raises
    ------
    ValueError
        If the file does not start with the NT-MDT signature, or if a frame is inconsistent.
    OSError
        If the file cannot be read.
    """
    buffer = Path(file).read_bytes()

    if len(buffer) < _MDT_FILE_HEADER_SIZE or not buffer.startswith(MDT_MAGIC):
        msg = f"{file} is not an NT-MDT '.mdt' file (wrong signature)."
        raise ValueError(msg)

    (last_frame,) = struct.unpack_from("<H", buffer, _MDT_FRAME_COUNT_OFFSET)

    frames: list[MdtAnyFrame] = []
    offset = _MDT_FILE_HEADER_SIZE
    for index in range(last_frame + 1):
        if offset >= len(buffer):
            logger.warning(
                "MDT file declares %d frames but only %d could be read.", last_frame + 1, index
            )
            break
        frame, offset = _read_mdt_frame(buffer, offset, index)
        if frame is not None:
            frames.append(frame)

    return frames
