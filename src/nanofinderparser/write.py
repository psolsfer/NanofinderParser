r"""Write NanoFinder SMD and S1D files.

Both formats are made of two parts: an XML header, whose lines are separated by ``\\r\\n``,
immediately followed by a raw block of little-endian ``float32`` values holding every spectrum,
one after another. Only the root element and the meaning of the spectra differ: an SMD file
roots its header at ``<SCANDATA>`` and holds one spectrum per point of a spatial scan, while an
S1D file roots it at ``<SPECTRADATA>`` and holds one spectrum per member of a kinetic series
measured at a single point.

This module writes both parts from a :class:`~nanofinderparser.models.Mapping` or a
:class:`~nanofinderparser.models.KineticSeries`, so that data built in memory (see
:mod:`nanofinderparser.synthetic`) or read from an existing file can be stored as a file that
:func:`~nanofinderparser.load.load_smd` or :func:`~nanofinderparser.load.load_s1d` reads back.

Notes
-----
Only the fields modelled by :class:`~nanofinderparser.models.Mapping` and
:class:`~nanofinderparser.models.KineticSeries` survive a read-modify-write cycle. The many
instrument settings that the parser ignores (bleaching options, axis tables, DAC limits, ...)
are written with the plausible defaults of this module rather than with the values of the
original file.
"""

import logging
from pathlib import Path
from typing import Any, Final

import numpy as np
import xmltodict
from numpy.typing import NDArray

from nanofinderparser.models import (
    SMD_DATE_FORMAT,
    SMD_TIME_FORMAT,
    Axis,
    Channel,
    KineticSeries,
    Mapping,
)
from nanofinderparser.parsers import S1D_ROOT, SMD_DTYPE, SMD_ROOT
from nanofinderparser.utils import format_vb_bool

logger = logging.getLogger(__name__)

# What this module can write: the two containers built on the same XML header.
type ScannedFrame = Mapping | KineticSeries

# Values written for the header fields that the parser does not model.
DEFAULT_VENDOR: Final[str] = "NNFinder"

# GUIDs identify the hardware of the instrument the file came from, so files written from
# scratch get the nil UUID instead of the identifier of anyone's setup. A mapping read from a
# real file keeps whatever GUID that file carried.
NIL_GUID: Final[str] = "{00000000-0000-0000-0000-000000000000}"

_AXIS_COUNT_MAX: Final[int] = 65535
_AXIS_COUNT_MIN: Final[int] = 0

# Line separator of the XML header, as written by NanoFinder.
_NEWLINE: Final[str] = "\r\n"

# The parser reads the header line by line, so a value may not contain a line break.
_LINE_BREAKS: Final[tuple[str, ...]] = ("\r\n", "\r", "\n")


def _text(value: Any) -> str:
    """Format a value the way NanoFinder writes it in the XML header.

    Parameters
    ----------
    value : Any
        The value to format. Booleans follow the Visual Basic convention (``-1`` / ``0``),
        integral floats lose their decimal part, and any other float keeps the shortest
        representation that reads back as the same number.

    Returns
    -------
    str
        The formatted value, free of line breaks.
    """
    if isinstance(value, bool):
        return format_vb_bool(value)
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        number = float(value)
        return str(int(number)) if number.is_integer() else repr(number)

    text = str(value)
    for break_ in _LINE_BREAKS:
        text = text.replace(break_, " ")
    return text


def _axis_dict(axis: Axis, size: int) -> dict[str, str]:
    """Build the XML block of a single stage axis.

    Parameters
    ----------
    axis : Axis
        The axis to write.
    size : int
        Number of points scanned along the axis, used to compute ``AxisCountStop``.

    Returns
    -------
    dict[str, str]
        The ``AxisX`` / ``AxisY`` / ``AxisZ`` block, ready to be serialized.
    """
    count_stop = axis.count_start + axis.count_step * max(size - 1, 0)
    return {
        "AxisIsPresent": _text(True),
        "AxisIsInUse": _text(axis.is_in_use),
        "AxisIsInversed": _text(axis.is_inversed),
        "AxisIsSlow": _text(axis.is_slow),
        "AxisName": _text(axis.name),
        "AxisMaker": "XYZ NI-DACmx",
        "AxisUnitName": _text(axis.unit_name),
        "AxisGuid": NIL_GUID,
        "AxisCountMax": _text(_AXIS_COUNT_MAX),
        "AxisCountMin": _text(_AXIS_COUNT_MIN),
        "AxisCountCurr": _text(axis.count_start),
        "AxisCountPrev": _text(axis.count_start),
        "AxisCountStart": _text(axis.count_start),
        "AxisCountStop": _text(count_stop),
        "AxisCountStep": _text(axis.count_step),
        "AxisUnitIndInteger": _text(-1),
        "AxisBiasFloat": _text(axis.bias_float),
        "AxisScaleFloat": _text(axis.scale_float),
        "AxisFloatPrecition": _text(3),
        "AxisCountLimMax": _text(_AXIS_COUNT_MAX),
        "AxisCountLimMin": _text(_AXIS_COUNT_MIN),
    }


def _channel_dict(channel: Channel) -> dict[str, Any]:
    """Build the XML block describing one detector channel.

    Parameters
    ----------
    channel : Channel
        The channel to write.

    Returns
    -------
    dict[str, Any]
        The ``Channel0`` block, ready to be serialized.
    """
    info_items = {
        f"Item{index}": _text(item) for index, item in enumerate(channel.channel_info.to_items())
    }
    axis_array = " ".join(_text(float(value)) for value in channel.channel_axis_array)
    return {
        "DeviceGuid": channel.device_guid or NIL_GUID,
        "DeviceName": _text(channel.device_name),
        "DeviceChannels": _text(1),
        "DeviceChannel": _text(0),
        "DataChannelName": _text(channel.data_channel_name),
        "DataChannelUnit": _text(channel.data_channel_unit),
        "ChannelType": _text(1),
        "ChannelSize": _text(channel.channel_size),
        "ChannelAxisName": _text(channel.channel_axis_name),
        "ChannelAxisUnit": _text(channel.channel_axis_unit),
        "ChannelAxisLaserWl": _text(channel.channel_axis_laser_wl),
        "SeriesType": _text(0),
        "SeriesSize": _text(channel.series_size),
        "SeriesAxisName": "Series",
        "SeriesAxisUnit": "None",
        "SeriesAxisLaserWl": _text(0),
        "ChannelAxisArray": f"{axis_array} ",
        # The series axis simply numbers the acquisitions, from one.
        "SeriesAxisArray": "".join(f"{index} " for index in range(1, channel.series_size + 1)),
        "ChannelInfoSize": _text(len(info_items)),
        "ChannelInfo": info_items,
    }


def _scandata_dict(
    frame: ScannedFrame,
    data_block_size_bytes: int,
    file: Path,
    root: str = SMD_ROOT,
) -> dict[str, Any]:
    """Build the whole XML header of an SMD or S1D file.

    Parameters
    ----------
    frame : Mapping | KineticSeries
        The mapping or kinetic series to describe.
    data_block_size_bytes : int
        Size of the binary block that follows the header.
    file : Path
        Destination file, used as ``OriginalFileName`` when `frame` does not carry one.
    root : str, optional
        Name of the root element, by default ``"SCANDATA"`` (an SMD file). A kinetic series is
        written under ``"SPECTRADATA"`` instead.

    Returns
    -------
    dict[str, Any]
        A single-rooted dictionary, ready for :func:`xmltodict.unparse`.
    """
    parameters = frame.scanned_frame_parameters
    header = parameters.frame_header
    options = parameters.frame_options
    stage = parameters.stage_3d_parameters
    axes = stage.stage_axes_dimensions
    calibration = parameters.data_calibration

    channels = {
        f"Channel{index}": _channel_dict(channel)
        for index, channel in enumerate(calibration.channels)
    }

    return {
        root: {
            "Vendor": frame.vendor or DEFAULT_VENDOR,
            "Version": frame.version or "1",
            "ScannedFrameParameters": {
                "Vendor": parameters.vendor,
                "Version": parameters.version,
                "ScanRepeatNumber": _text(parameters.scan_repeat_number),
                "FrameHeader": {
                    "Vendor": header.vendor,
                    "Version": header.version,
                    "Date": header.date_model.strftime(SMD_DATE_FORMAT),
                    "Time": header.time_model.strftime(SMD_TIME_FORMAT),
                    "Information": _text(header.information),
                    "SystemName": _text(header.system_name),
                    "PositioningSysName": _text(header.positioning_sys_name),
                    "DetectionSysName": _text(header.detection_sys_name),
                    "ScannedDataName": _text(header.scanned_data_name),
                    "FunctionName": "Not specified",
                    "Information1": "Not specified",
                    "Information2": "Not specified",
                    "Information3": "Not specified",
                    "Information4": "Not specified",
                    "Information5": "Not specified",
                },
                "FrameOptions": {
                    "Vendor": options.vendor,
                    "Version": options.version,
                    "ScanModel": _text(0),
                    "PointsMode": _text(0),
                    "DerectionMode": _text(0),
                    "DisplayResult": _text(2),
                    "Display3D": _text(1),
                    "StoreMode": _text(1),
                    "BleachEnable": _text(-1),
                    "BleachMode": _text(0),
                    "RepeatScan": _text(1),
                    "SwitchingOMUCfg": _text(0),
                    "OpenLoopX": _text(1),
                    "OpenLoopY": _text(0),
                    "OpenLoopZ": _text(0),
                    "OpenLoopQ": _text(0),
                    "MultiDetectionEnable": _text(0),
                    "MultiDetectionCount": _text(0),
                    "MultiDetectionMode": _text(0),
                    "LaserSpotShiftEnable": _text(0),
                    "LaserSpotShift_dX": _text(0),
                    "LaserSpotShift_dY": _text(0),
                    "LaserSpotShift_InvX": _text(0),
                    "LaserSpotShift_InvY": _text(0),
                    "RepeatDelayMSec": _text(0),
                    "StartDelayMSec": _text(0),
                    "IsAxisPointsByTableX": _text(0),
                    "IsAxisPointsByTableY": _text(0),
                    "IsAxisPointsByTableZ": _text(0),
                    "AxisPointsTableSzX": _text(0),
                    "AxisPointsTableSzY": _text(0),
                    "AxisPointsTableSzZ": _text(0),
                    "OmuLaserWLnm": _text(options.laser_wavelength_nm),
                    "OmuCurPower": _text(options.current_power),
                    "OmuGratingGroove": _text(options.grating_groove),
                    "OmuCentralWaveLengthNM": _text(options.central_wavelength_nm),
                    "OmuPinHoleSize": _text(options.pinhole_size),
                    "OmuHalfWavePos": _text(1),
                    "OmuBeamExpPos": _text(1),
                },
                "Stage3DParameters": {
                    "Vendor": stage.vendor,
                    "Version": stage.version,
                    "AxisCurrent": _text(0),
                    "AxisSizeX": _text(stage.axis_size_x),
                    "AxisSizeY": _text(stage.axis_size_y),
                    "AxisSizeZ": _text(stage.axis_size_z),
                    "StageAxesDimentions": {
                        "AxisX": _axis_dict(axes.x, stage.axis_size_x),
                        "AxisY": _axis_dict(axes.y, stage.axis_size_y),
                        "AxisZ": _axis_dict(axes.z, stage.axis_size_z),
                    },
                    "StageAxesDimentionTables": {
                        "AxisTableX": {"AxisTableSz": _text(0)},
                        "AxisTableY": {"AxisTableSz": _text(0)},
                        "AxisTableZ": {"AxisTableSz": _text(0)},
                    },
                },
                "DataCalibration": {
                    "Vendor": calibration.vendor,
                    "Version": calibration.version,
                    "Channels": _text(len(calibration.channels)),
                    "Channel": _text(0),
                    # Index of the acquisition the instrument was left showing: the last one.
                    "SeriesCur": _text(max(calibration.channels[0].series_size - 1, 0)),
                    "ChannelRmnInd": _text(0),
                    "SeriesRmnInd": _text(2),
                    "DataDimentions": channels,
                },
                "OriginalFileName": _text(parameters.original_file_name or file),
                "DataLocation": "Self",
                "DataBlockSizeBytes": _text(data_block_size_bytes),
            },
        }
    }


def _validate(frame: ScannedFrame, described_as: str) -> NDArray[np.float32]:
    """Check that the data of a mapping or a kinetic series matches what its header declares.

    Parameters
    ----------
    frame : Mapping | KineticSeries
        The mapping or kinetic series about to be written.
    described_as : str
        How the header describes the expected values, quoted in the error.

    Returns
    -------
    NDArray[np.float32]
        The flat data, as the little-endian ``float32`` values of the file.

    Raises
    ------
    NotImplementedError
        If the frame holds anything other than exactly one detector channel, or a layout the
        format does not support.
    ValueError
        If the number of values does not match what the header declares.
    """
    channel = frame.single_channel()

    if channel.channel_size != channel.channel_axis_array.size:
        msg = (
            f"The channel declares {channel.channel_size} points per spectrum but its spectral "
            f"axis holds {channel.channel_axis_array.size} values."
        )
        raise ValueError(msg)

    expected = frame.expected_data_size
    data = np.asarray(frame.data, dtype=SMD_DTYPE).ravel()
    if data.size != expected:
        msg = (
            f"The data holds {data.size} values, but its parameters describe {expected} "
            f"({described_as})."
        )
        raise ValueError(msg)

    return data


def _header(
    frame: ScannedFrame, data_block_size_bytes: int, file: Path, root: str = SMD_ROOT
) -> str:
    r"""Build the XML header of a file, with the line breaks NanoFinder writes.

    Parameters
    ----------
    frame : Mapping | KineticSeries
        The mapping or kinetic series to describe.
    data_block_size_bytes : int
        Size of the binary block that follows the header.
    file : Path
        Destination file, used as ``OriginalFileName`` when `frame` does not carry one.
    root : str, optional
        Name of the root element, by default ``"SCANDATA"`` (an SMD file).

    Returns
    -------
    str
        The header, from the XML declaration to the closing tag of `root`, ending in a newline.
    """
    xml: str = xmltodict.unparse(
        _scandata_dict(frame, data_block_size_bytes, file, root),
        pretty=True,
        indent="  ",
        full_document=False,
    )
    document = '<?xml version="1.0"?>\n' + xml + "\n"
    return document.replace("\n", _NEWLINE)


def _write(frame: ScannedFrame, file: Path, data: NDArray[np.float32], root: str) -> Path:
    """Write the XML header and the binary block of a file.

    Parameters
    ----------
    frame : Mapping | KineticSeries
        The mapping or kinetic series to write.
    file : Path
        Path of the file to write. Parent directories are created when missing.
    data : NDArray[np.float32]
        The validated flat data of `frame`.
    root : str
        Name of the root element of the XML header.

    Returns
    -------
    Path
        The path of the file just written.

    Raises
    ------
    OSError
        If the file cannot be written.
    """
    header = _header(frame, data.nbytes, file, root)

    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("wb") as stream:
        stream.write(header.encode("utf-8"))
        stream.write(data.tobytes())

    return file


def write_smd(mapping: Mapping, file: Path | str) -> Path:
    """Write a mapping as a NanoFinder SMD file.

    The file holds the XML header describing the scan followed by the raw ``float32`` block of
    every spectrum, exactly as the instrument writes them, so it can be read back with
    :func:`~nanofinderparser.load.load_smd`.

    Parameters
    ----------
    mapping : Mapping
        The mapping to write. Its data is stored as ``float32``, which is what SMD files hold,
        so writing a mapping whose data is ``float64`` loses precision.
    file : Path | str
        Path of the file to write. Parent directories are created when missing.

    Returns
    -------
    Path
        The path of the file just written.

    Raises
    ------
    NotImplementedError
        If the mapping holds more than one detector channel, or more than one acquisition per
        spatial point.
    ValueError
        If the number of values does not match the grid and spectrum length of the header.
    OSError
        If the file cannot be written.

    Examples
    --------
    >>> from nanofinderparser import load_smd, write_smd
    >>> mapping = load_smd(Path("original.smd"))  # doctest: +SKIP
    >>> write_smd(mapping, Path("copy.smd"))  # doctest: +SKIP

    """
    file = Path(file)
    channel = mapping.single_channel()
    x_steps, y_steps, z_steps = mapping.map_steps
    data = _validate(
        mapping, f"{x_steps} x {y_steps} x {z_steps} points of {channel.channel_size} each"
    )

    _write(mapping, file, data, SMD_ROOT)

    logger.debug("Wrote %d spectra to %s.", data.size // mapping.get_spectral_axis_len(), file)
    return file


def write_s1d(series: KineticSeries, file: Path | str) -> Path:
    """Write a kinetic series as a NanoFinder S1D file.

    The file holds the XML header describing the acquisition followed by the raw ``float32``
    block of every spectrum of the series, exactly as the instrument writes them, so it can be
    read back with :func:`~nanofinderparser.load.load_s1d`.

    Parameters
    ----------
    series : KineticSeries
        The kinetic series to write. Its data is stored as ``float32``, which is what S1D files
        hold, so writing a series whose data is ``float64`` loses precision.
    file : Path | str
        Path of the file to write. Parent directories are created when missing.

    Returns
    -------
    Path
        The path of the file just written.

    Raises
    ------
    NotImplementedError
        If the series holds more than one detector channel, or spans more than one stage point.
    ValueError
        If the number of values does not match the series and spectrum length of the header.
    OSError
        If the file cannot be written.

    Examples
    --------
    >>> from nanofinderparser import load_s1d, write_s1d
    >>> series = load_s1d(Path("original.s1d"))  # doctest: +SKIP
    >>> write_s1d(series, Path("copy.s1d"))  # doctest: +SKIP

    """
    file = Path(file)
    channel = series.single_channel()
    data = _validate(series, f"{channel.series_size} spectra of {channel.channel_size} points each")

    _write(series, file, data, S1D_ROOT)

    logger.debug("Wrote %d spectra to %s.", data.size // series.get_spectral_axis_len(), file)
    return file
