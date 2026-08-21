"""Models to parse nanofinder files."""

import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Literal, Protocol, Self, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyauxlib.fileutils.filesfolders import clean_filename
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nanofinderparser.map import AxisSpec, _nanofinder_mapcoords
from nanofinderparser.parsers import MdtImageFrame, MdtSpectrumFrame
from nanofinderparser.units import MdtUnit, Units, convert_spectral_units, validate_units
from nanofinderparser.utils import SaveMapCoords, validate_savemapcoords

logger = logging.getLogger(__name__)


class VendorVersion(BaseModel):
    """Base model for vendor and version information.

    Attributes
    ----------
    vendor : str
        The vendor name.
    version : str
        The version number or identifier.
    """

    vendor: str = Field(alias="Vendor")
    version: str = Field(alias="Version")


class FrameHeader(VendorVersion, BaseModel):
    """Model for frame header information.

    Attributes
    ----------
    date : date
        The date of the measurement.
    time : time
        The time of the measurement.
    information : str
        Additional information about the measurement.
    system_name : str
        The name of the system used for measurement.
    positioning_sys_name : str
        The name of the positioning system.
    detection_sys_name : str
        The name of the detection system.
    scanned_data_name : str
        The name of the scanned data.

    Methods
    -------
    datetime : datetime
        Property that combines date and time into a datetime object.
    """

    date_model: date = Field(alias="Date")
    time_model: time = Field(alias="Time")
    information: str = Field(alias="Information")
    system_name: str = Field(alias="SystemName")
    positioning_sys_name: str = Field(alias="PositioningSysName")
    detection_sys_name: str = Field(alias="DetectionSysName")
    scanned_data_name: str = Field(alias="ScannedDataName")  # IMPORTANT???

    @field_validator("date_model", mode="before")
    @classmethod
    def parse_date(cls, value: str) -> date:
        """To properly parse the date."""
        year, month, day = map(int, value.split("/"))
        return date(year, month, day)

    @field_validator("time_model", mode="before")
    @classmethod
    def parse_time(cls, value: str) -> time:
        """To properly parse the time."""
        hour, minute, second = map(int, value.split(":"))
        return time(hour, minute, second)

    @property
    def datetime(self) -> datetime:
        """Date and time of the measurement."""
        return datetime.combine(self.date_model, self.time_model)


class FrameOptions(VendorVersion, BaseModel):
    """Model for frame options.

    Attributes
    ----------
    laser_wavelength_nm : float
        The wavelength of the laser in nanometers.
    current_power : float
        The current power setting.
    grating_groove : str
        The grating groove information.
    central_wavelength_nm : float
        The central wavelength in nanometers.
    pinhole_size : float
        The size of the pinhole.
    """

    laser_wavelength_nm: float = Field(
        alias="OmuLaserWLnm"
    )  # IMPORTANT Wavelength of the laser in nm
    current_power: float = Field(alias="OmuCurPower")
    grating_groove: str = Field(alias="OmuGratingGroove")
    central_wavelength_nm: float = Field(
        alias="OmuCentralWaveLengthNM"
    )  # IMPORTANT Wavelength center in nm for the spectrum
    pinhole_size: float = Field(alias="OmuPinHoleSize")  # In micrometers


class Axis(BaseModel):
    """Model for axis information.

    Attributes
    ----------
    is_in_use : int
        Indicator if the axis is in use (NanoFinder VB convention: -1 = True, 0 = False).
    is_inversed : bool
        Whether the axis direction is inverted.
    is_slow : bool
        Whether this is the slow scan axis.
    name : str
        The name of the axis.
    unit_name : str
        The unit name for the axis.
    count_start : int
        Raw DAC count at the start of the scan.
    count_step : int
        Raw DAC count increment per step.
    bias_float : float
        Physical offset applied to all positions (in axis units).
    scale_float : float
        Conversion factor from raw DAC counts to physical units.

    Methods
    -------
    step_size : float
        Property that calculates the physical step size.
    start_position : float
        Property that returns the absolute physical start position.
    step_units : str
        Property that returns the step units.
    """

    is_in_use: int = Field(alias="AxisIsInUse")
    is_inversed: bool = Field(alias="AxisIsInversed")
    is_slow: bool = Field(alias="AxisIsSlow")
    name: str = Field(alias="AxisName")
    unit_name: str = Field(alias="AxisUnitName")  # IMPORTANT Units of the axis
    count_start: int = Field(alias="AxisCountStart")
    count_step: int = Field(alias="AxisCountStep")
    bias_float: float = Field(alias="AxisBiasFloat")
    scale_float: float = Field(alias="AxisScaleFloat")

    @field_validator("is_inversed", "is_slow", mode="before")
    @classmethod
    def parse_vb_bool(cls, value: str | bool | int) -> bool:
        """Parse Visual Basic boolean convention: '0'/0 → False, '-1'/-1 → True.

        Parameters
        ----------
        value : str | bool | int
            Raw value from the XML.

        Returns
        -------
        bool
            Parsed boolean value.

        Raises
        ------
        ValueError
            If a string value other than '0' or '-1' is encountered.
        """
        if isinstance(value, str):
            if value == "0":
                return False
            if value == "-1":
                return True
            msg = f"Unexpected boolean string value: {value!r}; expected '0' or '-1'"
            raise ValueError(msg)
        return bool(value)

    @property
    def step_size(self) -> float:
        """Physical step size in axis units: ``count_step * scale_float``."""
        return self.count_step * self.scale_float

    @property
    def start_position(self) -> float:
        """Absolute physical start position of the scan in axis units.

        Computed as ``bias_float + count_start * scale_float``.
        """
        return self.bias_float + self.count_start * self.scale_float

    @property
    def step_units(self) -> str:
        """Units of the step."""
        return self.unit_name


class StageAxesDimensions(BaseModel):
    """Model for stage axes dimensions.

    Attributes
    ----------
    x : Axis
        The X-axis information.
    y : Axis
        The Y-axis information.
    z : Axis
        The Z-axis information.

    Methods
    -------
    step_size : tuple[float, float, float]
        Property that returns the step sizes for all axes.
    step_units : tuple[str, str, str]
        Property that returns the step units for all axes.
    """

    x: Axis = Field(alias="AxisX")
    y: Axis = Field(alias="AxisY")
    z: Axis = Field(alias="AxisZ")

    @property
    def step_size(self) -> tuple[float, float, float]:
        """Size of the map steps in the (x,y,z) axes."""
        return (self.x.step_size, self.y.step_size, self.z.step_size)

    @property
    def step_units(self) -> tuple[str, str, str]:
        """Units of the map steps in the (x,y,z) axes."""
        return (self.x.step_units, self.y.step_units, self.z.step_units)

    @property
    def start_position(self) -> tuple[float, float, float]:
        """Absolute physical start position of the scan in the (x,y,z) axes."""
        return (self.x.start_position, self.y.start_position, self.z.start_position)


class Stage3DParameters(VendorVersion, BaseModel):
    """Model for 3D stage parameters.

    Attributes
    ----------
    axis_size_x : int
        The size (number of steps) of the X-axis.
    axis_size_y : int
        The size (number of steps) of the Y-axis.
    axis_size_z : int
        The size (number of steps) of the Z-axis.
    stage_axes_dimensions : StageAxesDimensions
        The dimensions of the stage axes.

    Methods
    -------
    map_steps : tuple[int, int, int]
        Property that returns the map steps for all axes.
    """

    axis_size_x: int = Field(alias="AxisSizeX")  # IMPORTANT Number of steps of mapping
    axis_size_y: int = Field(alias="AxisSizeY")  # IMPORTANT Number of steps of mapping
    axis_size_z: int = Field(alias="AxisSizeZ")
    stage_axes_dimensions: StageAxesDimensions = Field(alias="StageAxesDimentions")

    @property
    def map_steps(self) -> tuple[int, int, int]:
        """Number of steps of the map in the (x,y,z) axes."""
        return (self.axis_size_x, self.axis_size_y, self.axis_size_z)

    @property
    def scan_order(self) -> tuple[int, int]:
        """Return (slow_axis_steps, fast_axis_steps) for reshape into a 2-D map.

        Inferred from AxisIsSlow metadata. Falls back to (y_steps, x_steps) — i.e. x-fast — which is
        NanoFinder's default raster scan convention, consistent with the row ordering used in
        to_df().
        """
        axes = self.stage_axes_dimensions
        if axes.x.is_slow and not axes.y.is_slow:
            # x is slow axis: scan order is (x, y) → reshape (x_steps, y_steps)
            return self.axis_size_x, self.axis_size_y
        # y is slow axis (or ambiguous): default x-fast → reshape (y_steps, x_steps)
        return self.axis_size_y, self.axis_size_x


class ChannelInfo(BaseModel):
    """Model for detailed channel information.

    This class represents additional information about the channel, including CCD hardware
    details, acquisition settings, and readout parameters.

    Attributes
    ----------
    temperature : float | None
        CCD temperature in degrees Celsius.
    exposure_time : float | None
        Exposure time per acquisition in seconds.
    cycle_time : float | None
        Total cycle time per acquisition in seconds.
    acquisition_mode : str | None
        Acquisition mode (e.g., ``"accumulate"``, ``"single"``).
    accumulation_number : int | None
        Number of accumulations (only set when acquisition_mode is ``"accumulate"``).
    head_model : str | None
        CCD detector head model identifier.
    ccd_width : int | None
        Full CCD width in pixels.
    ccd_height : int | None
        Full CCD height in pixels.
    central_pixel : int | None
        Central pixel index of the CCD used for calibration.
    pixel_size_um : float | None
        Physical size of each CCD pixel in micrometers.
    horizontal_binning : int | None
        Horizontal pixel binning factor.
    center_row : int | None
        Center row of the readout track on the CCD.
    track_height : int | None
        Height (in rows) of the readout track on the CCD.
    readout_mode : str | None
        CCD readout mode (e.g., ``"Single Track"``).

    Notes
    -----
    All attributes are optional (can be ``None``) to accommodate varying levels of available
    information from different data sources.
    """

    temperature: float | None = Field(None, alias="Temperature")
    exposure_time: float | None = Field(None, alias="ExposureTime")
    cycle_time: float | None = Field(None, alias="CycleTime")
    acquisition_mode: str | None = Field(None, alias="AcquisitionMode")
    accumulation_number: int | None = Field(None, alias="AccumulationNumber")
    head_model: str | None = Field(None, alias="HeadModel")
    ccd_width: int | None = Field(None, alias="CcdWidth")
    ccd_height: int | None = Field(None, alias="CcdHeight")
    central_pixel: int | None = Field(None, alias="CentralPixel")
    pixel_size_um: float | None = Field(None, alias="PixelSizeUm")
    horizontal_binning: int | None = Field(None, alias="HorizontalBinning")
    center_row: int | None = Field(None, alias="CenterRow")
    track_height: int | None = Field(None, alias="TrackHeight")
    readout_mode: str | None = Field(None, alias="ReadoutMode")


class Channel(BaseModel):
    """Model for channel information.

    Attributes
    ----------
    device_guid : str
        The GUID of the device.
    device_name : str
        The name of the device.
    data_channel_name : str
        The name of the data channel.
    data_channel_unit : str
        The unit of the data channel.
    channel_size : int
        The size of the channel.
    channel_axis_name : str
        The name of the channel axis.
    channel_axis_unit : Literal["nm", "cm-1", "eV"]
        The unit of the channel axis.
    channel_axis_laser_wl : float
        The laser wavelength for the channel axis.
    channel_axis_array : NDArray[np.float64]
        The array of channel axis values.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # To allow having numpy arrays

    device_guid: str = Field(alias="DeviceGuid")
    device_name: str = Field(alias="DeviceName")
    data_channel_name: str = Field(alias="DataChannelName")  # For example "Photons"
    data_channel_unit: str = Field(alias="DataChannelUnit")  # For example "Counts"
    channel_size: int = Field(
        alias="ChannelSize"
    )  # IMPORTANT Number of data points of each spectrum
    channel_axis_name: str = Field(alias="ChannelAxisName")  # For example "Wavelength"
    channel_axis_unit: Literal["nm", "cm-1", "eV"] = Field(
        alias="ChannelAxisUnit"
    )  # IMPORTANT NanoFinder uses nm in SMD files
    channel_axis_laser_wl: float = Field(
        alias="ChannelAxisLaserWl"
    )  # IMPORTANT Wavelength excitation
    channel_axis_array: NDArray[np.float64] = Field(
        alias="ChannelAxisArray"
    )  # IMPORTANT # Spectral axis, with units given by ChannelAxisUnit
    series_size: int = Field(
        alias="SeriesSize"
    )  # Number of series per spatial point; >1 not yet supported

    channel_info: ChannelInfo = Field(alias="ChannelInfo")

    @field_validator("channel_axis_array", mode="before")
    @classmethod
    def parse_chanelaxisarray(cls, value: str) -> NDArray[np.float64]:
        """Properly parse the ChannelAxisArray."""
        return np.fromstring(value, sep=" ")

    @field_validator("channel_info", mode="before")
    @classmethod
    def parse_channel_info(cls, value: dict[str, str]) -> ChannelInfo:
        """Parse the ChannelInfo free-text items into a structured ChannelInfo object.

        Parameters
        ----------
        value : dict[str, str]
            Raw dict of ``{ItemN: "Key = Value"}`` strings from the XML.

        Returns
        -------
        ChannelInfo
            Populated ChannelInfo instance.
        """
        info_dict: dict[str, Any] = {}

        parsers: dict[str, tuple[str, Callable[[str], Any]]] = {
            "Head model": ("HeadModel", str),
            "CCD Width": ("CcdWidth", int),
            "CCD Height": ("CcdHeight", int),
            "Central Pixel": ("CentralPixel", int),
            "Pixel Size": ("PixelSizeUm", float),
            "Temperature": ("Temperature", float),
            "Horizontal binning": ("HorizontalBinning", int),
            "Center Row": ("CenterRow", int),
            "Track Height": ("TrackHeight", int),
        }

        for text in value.values():
            if cls._parse_simple_field(text, parsers, info_dict):
                continue

            if "Readout Mode" in text:
                info_dict["ReadoutMode"] = text.split(":", 1)[1].strip()
            elif "Acquisition mode" in text:
                cls._parse_acquisition_mode(text, info_dict)
            elif "Exposure time" in text:
                cls._parse_exposure_time(text, info_dict)

        return ChannelInfo(**info_dict)

    @staticmethod
    def _parse_simple_field(
        text: str,
        parsers: dict[str, tuple[str, Callable[[str], Any]]],
        info: dict[str, Any],
    ) -> bool:
        for key, (field, converter) in parsers.items():
            if key in text:
                info[field] = converter(text.split("=", 1)[1].strip())
                return True

        return False

    @staticmethod
    def _parse_acquisition_mode(
        text: str,
        info: dict[str, Any],
    ) -> None:
        mode_info = text.split(":", 1)[1].strip().split(".")
        mode = mode_info[0].strip().lower()

        info["AcquisitionMode"] = mode

        if mode in {"accomulate", "accumulate"}:  # NOTE There's a typo in smd files
            info["AccumulationNumber"] = int(mode_info[1].split("=", 1)[1].strip())

    @staticmethod
    def _parse_exposure_time(
        text: str,
        info: dict[str, Any],
    ) -> None:
        parts = text.split("=")

        info["ExposureTime"] = float(parts[1].strip().split()[0])
        info["CycleTime"] = float(parts[2].strip())

    @staticmethod
    def _parse_string_field(
        field: str,
    ) -> Callable[[str, dict[str, Any]], None]:
        def parser(text: str, info: dict[str, Any]) -> None:
            info[field] = text.split("=", 1)[1].strip()

        return parser

    @staticmethod
    def _parse_int_field(
        field: str,
    ) -> Callable[[str, dict[str, Any]], None]:
        def parser(text: str, info: dict[str, Any]) -> None:
            info[field] = int(text.split("=", 1)[1].strip())

        return parser

    @staticmethod
    def _parse_float_field(
        field: str,
    ) -> Callable[[str, dict[str, Any]], None]:
        def parser(text: str, info: dict[str, Any]) -> None:
            info[field] = float(text.split("=", 1)[1].strip())

        return parser

    @staticmethod
    def _parse_colon_field(
        field: str,
    ) -> Callable[[str, dict[str, Any]], None]:
        def parser(text: str, info: dict[str, Any]) -> None:
            info[field] = text.split(":", 1)[1].strip()

        return parser


class DataCalibration(VendorVersion):
    """Model for data calibration information.

    Attributes
    ----------
    channels : List[Channel]
        List of channels in the data calibration.
    """

    channels: list[Channel] = Field(alias="Channels", default_factory=list)


class ScannedFrameParameters(VendorVersion, BaseModel):
    """Model for scanned frame parameters.

    Attributes
    ----------
    scan_repeat_number : int
        The number of scan repeats.
    frame_header : FrameHeader
        The frame header information.
    frame_options : FrameOptions
        The frame options.
    stage_3d_parameters : Stage3DParameters
        The 3D stage parameters.
    data_calibration : DataCalibration
        The data calibration information.
    original_file_name : str | None
        Original file path as stored on the acquisition computer, if present.
    data_block_size_bytes : int | None
        Expected size of the binary data block in bytes, if present.
    """

    scan_repeat_number: int = Field(alias="ScanRepeatNumber")
    frame_header: FrameHeader = Field(alias="FrameHeader")
    frame_options: FrameOptions = Field(alias="FrameOptions")
    stage_3d_parameters: Stage3DParameters = Field(alias="Stage3DParameters")
    data_calibration: DataCalibration = Field(alias="DataCalibration")
    original_file_name: str | None = Field(None, alias="OriginalFileName")
    data_block_size_bytes: int | None = Field(None, alias="DataBlockSizeBytes")


class Mapping:
    """Model for the complete mapping data obtained from a .smd file.

    This class represents the mapping data from a NanoFinder .smd file, including
    scanned frame parameters and the actual spectral data.

    Note: It is recommended to create instances of this class using the `load_smd`
    function rather than instantiating it directly.

    Attributes
    ----------
    vendor : str
        The vendor of the data.
    version : str
        The version of the data format.
    scanned_frame_parameters : ScannedFrameParameters
        The scanned frame parameters.
    data : NDArray
        The raw flat spectral data as read from the binary section of the SMD file, shape
        ``(n_spectra * spectral_len,)``.

    Properties
    ----------
    laser_wavelength : float
        Wavelength of the laser in nm.
    laser_power : float
        Power of the laser in mW.
    datetime : datetime
        Date and time of the measurement.
    date : date
        Date of the measurement.
    original_file_name : str | None
        Original file path as stored on the acquisition computer.
    step_size : tuple[float, float, float]
        Size of the map steps in the (x,y,z) axes.
    step_units : tuple[str, str, str]
        Units of the map steps in the (x,y,z) axes.
    map_steps : tuple[int, int, int]
        Number of steps of the map in the (x,y,z) axes.
    map_start : tuple[float, float, float]
        Absolute physical start position of the scan in the (x,y,z) axes.
    map_size : tuple[float, float, float]
        Size of the map in the (x,y,z) axes, with the corresponding units for each axis.

    Methods
    -------
    get_spectra(channel: int = 0)
        Return data reshaped as (n_spectra, spectral_len).
    get_spectral_axis(channel: int = 0)
        Get the spectral axis for the given channel.
    get_spectral_axis_len(channel: int = 0)
        Get the number of data points of each spectrum for the given channel.
    get_exposure_time(channel: int = 0)
        Get the exposure time of the given channel.
    get_accumulation_number(channel: int = 0)
        Get the accumulation number of the given channel.
    _get_data_to_map(channel: int = 0)
        Reshape the data as the mapping: (x, y, spectrum) for the given channel.
    _get_channel_axis_unit(channel: int = 0)
        Get the units of the spectral axis for the given channel.
    to_csv(path: Path = Path(), filename: str = "",
            spectral_units: Units | str | None = None,
            save_mapcoords: bool = False, channel: int = 0)
        Export the data to csv files.
    to_df(spectral_units: Units | str | None = None, channel: int = 0)
        Export the data and mapcoords to DataFrames.

    Notes
    -----
    # TODO
    Currently, some methods that accept a 'channel' parameter default to 'channel = 0'. At present,
    we don't have SMD files with multiple channels, so it's not yet clear how to handle them
    properly. Until we encounter multi-channel SMD files, keep using 'channel = 0' for all
    operations.
    """

    def __init__(self, init_dict: dict[Any, Any]) -> None:
        """Initialize a Mapping instance.

        Parameters
        ----------
        init_dict : dict[str, Any]
            A dictionary containing the initialization data for the Mapping instance.
            Expected keys:
            - 'Vendor': str, optional
            - 'Version': str, optional
            - 'ScannedFrameParameters': dict
            - 'Data': list[float]

        Raises
        ------
        KeyError
            If any of the required keys are missing from init_dict.
        """
        self.vendor: str = init_dict.get("Vendor", "")
        self.version: str = init_dict.get("Version", "")
        self.scanned_frame_parameters = ScannedFrameParameters(
            **init_dict["ScannedFrameParameters"]
        )
        self.data = init_dict["Data"]

    @property
    def data(self) -> NDArray[Any]:
        """The raw flat array of all spectral data as read from the binary part of the SMD file.

        The array has shape ``(n_spectra * spectral_len,)`` — i.e. it is stored exactly as it comes
        out of the binary section, without any reshaping.  Use :meth:`get_spectra` to obtain a 2-D
        ``(n_spectra, spectral_len)`` view, or :meth:`_get_data_to_map` for the full
        ``(slow_axis, fast_axis, spectral_len)`` spatial map.
        """
        return self._data

    @data.setter
    def data(self, value: list[float]) -> None:
        self._data = np.asarray(value)  # dtype inferred; stored flat as-is

    def get_spectra(self, channel: int = 0) -> NDArray[Any]:
        """Return the spectral data reshaped as ``(n_spectra, spectral_len)``.

        This is the canonical 2-D view of the flat :attr:`data` array: one row per spatial point,
        one column per spectral channel.  The row order matches the acquisition order produced by
        :func:`~nanofinderparser.map._nanofinder_mapcoords` (x-fast raster by default).

        Parameters
        ----------
        channel : int, optional
            The channel index, by default 0.

        Returns
        -------
        NDArray[Any]
            Array of shape ``(n_spectra, spectral_len)``.

        Notes
        -----
        # TODO
        Currently only ``channel = 0`` is supported.  Multi-channel SMD files are not yet
        encountered, so the reshape logic assumes a single contiguous data block.
        """
        return self._data.reshape(-1, self.get_spectral_axis_len(channel=channel))

    def get_spectral_axis(
        self,
        spectral_units: Units | Literal["nm", "cm-1", "eV", "raman_shift"] | None = None,
        channel: int = 0,
    ) -> NDArray[np.float64]:
        """Get the spectral axis for the given channel, optionally converting to specified units.

        Parameters
        ----------
        spectral_units : Units | {"nm", "cm-1", "eV", "raman_shift"} | None, optional
            The units to convert the spectral axis to. If None, returns the original units.
        channel : int, optional
            The channel index, by default 0

        Returns
        -------
        NDArray[np.float64]
            Array containing the spectral axis for the given channel, in the specified units.
        """
        raw_axis = self._get_raw_spectral_axis(channel)
        current_units = self._get_channel_axis_unit(channel)
        if spectral_units is None or current_units == spectral_units:
            return raw_axis

        new_unit = validate_units(spectral_units)

        return convert_spectral_units(
            raw_axis,
            self._get_channel_axis_unit(channel),
            new_unit,
            laser_wavelength_nm=self.laser_wavelength,
        )

    def get_spectral_axis_len(self, channel: int = 0) -> int:
        """Get the number of data points of each spectrum for the given channel.

        Parameters
        ----------
        channel : int, optional
            The channel index, by default 0

        Returns
        -------
        int
            Number of data points of each spectrum.
        """
        channel_obj = self.scanned_frame_parameters.data_calibration.channels[channel]
        return channel_obj.channel_size

    def _get_raw_spectral_axis(self, channel: int = 0) -> NDArray[np.float64]:
        """Get the raw spectral axis for the given channel without unit conversion.

        Parameters
        ----------
        channel : int, optional
            The channel index, by default 0

        Returns
        -------
        NDArray[np.float64]
            Array containing the raw spectral axis for the given channel.
        """
        channel_obj = self.scanned_frame_parameters.data_calibration.channels[channel]
        return channel_obj.channel_axis_array

    def get_exposure_time(self, channel: int = 0) -> float | None:
        """Get the exposure time the given channel.

        Parameters
        ----------
        channel : int, optional
            The channel index, by default 0

        Returns
        -------
        float | None
            Exposure time for the given channel.
        """
        channel_obj = self.scanned_frame_parameters.data_calibration.channels[channel]
        return channel_obj.channel_info.exposure_time

    def get_accumulation_number(self, channel: int = 0) -> int | None:
        """Get the accumulation number of the given channel.

        Parameters
        ----------
        channel : int, optional
            The channel index, by default 0

        Returns
        -------
        int | None
            Accumulation number for the given channel.
        """
        channel_obj = self.scanned_frame_parameters.data_calibration.channels[channel]
        return channel_obj.channel_info.accumulation_number

    @property
    def laser_wavelength(self) -> float:
        """Wavelength of the laser in nm."""
        return self.scanned_frame_parameters.frame_options.laser_wavelength_nm

    @property
    def laser_power(self) -> float:
        """Power of the laser in mW."""
        return self.scanned_frame_parameters.frame_options.current_power

    @property
    def datetime(self) -> datetime:
        """Date and time of the measurement."""
        return self.scanned_frame_parameters.frame_header.datetime

    @property
    def date(self) -> date:
        """Date of the measurement."""
        return self.scanned_frame_parameters.frame_header.date_model

    @property
    def step_size(self) -> tuple[float, float, float]:
        """Size of the map steps in the (x,y,z) axes."""
        return self.scanned_frame_parameters.stage_3d_parameters.stage_axes_dimensions.step_size

    @property
    def step_units(self) -> tuple[str, str, str]:
        """Units of the map steps in the (x,y,z) axes."""
        return self.scanned_frame_parameters.stage_3d_parameters.stage_axes_dimensions.step_units

    @property
    def map_steps(self) -> tuple[int, int, int]:
        """Number of steps of the map in the (x,y,z) axes."""
        return self.scanned_frame_parameters.stage_3d_parameters.map_steps

    @property
    def original_file_name(self) -> str | None:
        """Original file path as stored on the acquisition computer."""
        return self.scanned_frame_parameters.original_file_name

    @property
    def map_start(self) -> tuple[float, float, float]:
        """Absolute physical start position of the scan in the (x,y,z) axes."""
        return (
            self.scanned_frame_parameters.stage_3d_parameters.stage_axes_dimensions.start_position
        )

    @property
    def map_size(self) -> tuple[float, float, float]:
        """Size of the map in the (x,y,z) axes, with the corresponding units for each axis."""
        return (
            self.step_size[0] * (self.map_steps[0] - 1),
            self.step_size[1] * (self.map_steps[1] - 1),
            self.step_size[2] * (self.map_steps[2] - 1),
        )

    def _get_data_to_map(self, channel: int = 0) -> NDArray[Any]:
        """Reshape the data as a 3-D spatial map: ``(slow_axis, fast_axis, spectral_len)``.

        The slow/fast axis order is inferred from the ``AxisIsSlow`` metadata in the SMD file.
        For the default NanoFinder x-fast raster scan (both axes report ``AxisIsSlow=0``), this
        returns shape ``(y_steps, x_steps, spectral_len)``.

        Parameters
        ----------
        channel : int, optional
            The channel index, by default 0.

        Returns
        -------
        NDArray[Any]
            Array of shape ``(slow_steps, fast_steps, spectral_len)``.
        """
        slow, fast = self.scanned_frame_parameters.stage_3d_parameters.scan_order
        return self._data.reshape((slow, fast, self.get_spectral_axis_len(channel)))

    def _get_channel_axis_unit(self, channel: int = 0) -> Literal["nm", "cm-1", "eV"]:
        """Get the units of the spectral axis for the given channel.

        Parameters
        ----------
        channel : int, optional
            The channel index, by default 0

        Returns
        -------
        Literal["nm", "cm-1", "eV"]
            Units of the spectral axis.
        """
        channel_obj = self.scanned_frame_parameters.data_calibration.channels[channel]
        return channel_obj.channel_axis_unit

    def to_csv(
        self,
        path: Path = Path(),
        filename: str = "",
        spectral_units: Units | Literal["nm", "cm-1", "eV", "raman_shift"] | None = None,
        save_mapcoords: SaveMapCoords | str = SaveMapCoords.combined,
        channel: int = 0,
    ) -> None:
        """Export the data to csv files.

        It exports the data of the spectra and the mapping coordinates to their respective files.
        For the data, the header corresponds to the spectral axis in the selected units, and each
        row corresponds to a single spectrum.
        Each row of the mapping coordinates file provides the coordinates of the spectrum in the
        same row.

        Parameters
        ----------
        path : Path, optional
            Folder in which the files will be saved, by default Path()
        filename : str, optional
            Suffix to use for the name of the files, by default "".
            Any extension will be removed.
        spectral_units : Units | {"nm", "cm-1", "eV", "raman_shift"} | None, optional
            Units in which the spectral axis will be exported, by default None
        save_mapcoords : SaveMapCoords or {"no", "combined", "separated"}, optional
            How to save the mapping coordinates:
            - "no": Don't save mapping coordinates
            - "combined": Save mapping coordinates in the same file as the data
            - "separated": Save mapping coordinates in a separate file
            By default SaveMapCoords.combined.
        channel : int, optional
            The channel index to export, by default 0
        """
        save_mapcoords = validate_savemapcoords(save_mapcoords)

        if spectral_units is not None:
            spectral_units = validate_units(spectral_units)

        data, mapcoords = self.to_df(spectral_units, channel=channel)

        if not filename:
            map_file_path = path / "data.csv"
            coord_file_path = path / "mapcoords.csv"
        else:
            filename = Path(filename).with_suffix("").as_posix()
            if save_mapcoords == "separated":
                map_file_path = path / (filename + "_data.csv")
            else:
                map_file_path = path / (filename + ".csv")
            coord_file_path = path / (filename + "_mapcoords.csv")

        index = save_mapcoords not in ["separated", "no"]

        path.mkdir(parents=True, exist_ok=True)
        data.to_csv(map_file_path, na_rep="NaN", index=index)
        if save_mapcoords == "separated":
            mapcoords.to_csv(coord_file_path, na_rep="NaN", index=False)

    def to_df(
        self,
        spectral_units: Units | Literal["nm", "cm-1", "eV", "raman_shift"] | None = None,
        index: Literal["mapcoords", False] = "mapcoords",
        channel: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export the data and mapcoords to DataFrames.

        It exports the data of the spectra and the mapping coordinates as pandas DataFrames.
        For the data, the header corresponds to the spectral axis in the selected units, and each
        row corresponds to a single spectrum.
        The DataFrame for the mapping coordinates is aligned with that of the data: each row of the
        mapping coordinates provides the coordinates of the spectrum in the same row of the data.

        Parameters
        ----------
        spectral_units : Units | {"nm", "cm-1", "eV", "raman_shift"} | None, optional
            Units in which the spectral axis will be exported, by default None
        channel : int, optional
            The channel index to export, by default 0
        index : Literal["mapcoords", False], optional
            Use the mapping coordinates as index of the df, by default "mapcoords"

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            The data and mapping coordinates as DataFrames. The mapping coordinates' 'x' and 'y'
            columns carry `pint` units (via `pint-pandas`) matching the stage axes' units, when
            available.
        """
        spectral_axis = self.get_spectral_axis(spectral_units=spectral_units, channel=channel)

        # TODO only 2D (x and y) maps are supported for now; z-axis and true 3D maps would need a
        # different coordinate generation strategy.
        axes = self.scanned_frame_parameters.stage_3d_parameters.stage_axes_dimensions
        mapcoords = _nanofinder_mapcoords(
            self.map_steps[0],
            self.map_steps[1],
            x_axis=AxisSpec(axes.x.start_position, axes.x.step_size, axes.x.unit_name),
            y_axis=AxisSpec(axes.y.start_position, axes.y.step_size, axes.y.unit_name),
        )

        data = pd.DataFrame(
            self.get_spectra(channel),
            columns=spectral_axis,
            index=pd.MultiIndex.from_arrays([mapcoords["x"], mapcoords["y"]]),
        )

        # Reordering the rows
        # NOTE: this is not essential, only done to coincide with NanoFinder's convention of 'y'
        # starting from the bottom side of the mapping area
        mapcoords = mapcoords.sort_values(by=["y", "x"], ascending=[False, True])
        data = data.reindex(mapcoords.set_index(["x", "y"]).index)

        mapcoords = mapcoords.reset_index(drop=True)
        if not index:
            data = data.reset_index(drop=True)

        return data, mapcoords


# Number of arrays NanoFinder stores per spectrum frame: the spectral axis and the intensities.
_MDT_SPECTRUM_ARRAY_COUNT: int = 2


@dataclass(frozen=True, slots=True, eq=False)
class Spectrum:
    """A single spectrum, as stored in a NanoFinder ``.mdt`` file.

    Unlike :class:`Mapping`, which holds one spectrum per point of a spatial scan, a Spectrum is a
    standalone measurement with no associated stage coordinates. A ``.mdt`` file usually holds
    several of them; use :func:`~nanofinderparser.load.load_mdt` to obtain a :class:`Spectra`
    collection rather than building instances directly.

    Attributes
    ----------
    title : str
        Name given to the spectrum in NanoFinder, for example ``"Spectrum_1"``.
    spectral_axis : NDArray[np.float64]
        The spectral axis, in the units given by `spectral_axis_unit`.
    data : NDArray[np.float64]
        The measured intensities, aligned with `spectral_axis`.
    spectral_axis_unit : Units
        Units of the stored spectral axis. NanoFinder writes ``nm`` in ``.mdt`` files.
    data_unit : str
        Units of the intensities, typically ``"counts"``.
    laser_wavelength : float | None
        Excitation wavelength in nm, or None when the file does not record it.
    measured_at : datetime | None
        Acquisition timestamp, or None when the file stores an invalid date.
    text_comment : str
        Free-text comment attached to the measurement, empty when absent.

    Properties
    ----------
    datetime : datetime | None
        Date and time of the measurement.
    date : date | None
        Date of the measurement.
    spectral_axis_len : int
        Number of data points of the spectrum.

    Methods
    -------
    get_spectral_axis(spectral_units=None)
        Get the spectral axis, optionally converted to other units.
    to_df(spectral_units=None)
        Export the spectrum to a DataFrame indexed by the spectral axis.
    to_csv(path=Path(), filename="", spectral_units=None)
        Export the spectrum to a csv file.
    """

    title: str
    spectral_axis: NDArray[np.float64]
    data: NDArray[np.float64]
    spectral_axis_unit: Units
    data_unit: str
    laser_wavelength: float | None
    measured_at: datetime | None
    text_comment: str = ""

    @classmethod
    def from_mdt_frame(cls, frame: MdtSpectrumFrame) -> Self:
        """Build a Spectrum from a raw frame read out of a ``.mdt`` file.

        Parameters
        ----------
        frame : MdtSpectrumFrame
            The decoded frame, as returned by
            :func:`~nanofinderparser.parsers.read_mdt_frames`.

        Returns
        -------
        Spectrum
            The corresponding spectrum.

        Raises
        ------
        NotImplementedError
            If the frame stores a number of arrays other than two, which is not supported yet.
        ValueError
            If the spectral axis of the frame is stored in units that are not spectral.
        """
        if frame.array_count != _MDT_SPECTRUM_ARRAY_COUNT:
            msg = (
                f"MDT frame {frame.index} stores {frame.array_count} arrays; only frames with "
                f"{_MDT_SPECTRUM_ARRAY_COUNT} (spectral axis and intensities) are supported."
            )
            raise NotImplementedError(msg)

        return cls(
            title=frame.title,
            spectral_axis=frame.arrays[0],
            data=frame.arrays[1],
            spectral_axis_unit=MdtUnit(frame.x_scale.unit_code).spectral_units,
            data_unit=_mdt_unit_name(frame.z_scale.unit_code),
            laser_wavelength=frame.laser_wavelength_nm,
            measured_at=frame.measured_at,
            text_comment=frame.text_comment,
        )

    @property
    def datetime(self) -> datetime | None:
        """Date and time of the measurement."""
        return self.measured_at

    @property
    def date(self) -> date | None:
        """Date of the measurement."""
        return None if self.measured_at is None else self.measured_at.date()

    @property
    def spectral_axis_len(self) -> int:
        """Number of data points of the spectrum."""
        return int(self.spectral_axis.size)

    def get_spectral_axis(
        self,
        spectral_units: Units | Literal["nm", "cm-1", "eV", "raman_shift"] | None = None,
    ) -> NDArray[np.float64]:
        """Get the spectral axis, optionally converting it to the specified units.

        Parameters
        ----------
        spectral_units : Units | {"nm", "cm-1", "eV", "raman_shift"} | None, optional
            The units to convert the spectral axis to. If None, returns the original units.

        Returns
        -------
        NDArray[np.float64]
            The spectral axis in the specified units.

        Raises
        ------
        ValueError
            If a conversion to or from ``raman_shift`` is requested but the file does not record
            the excitation wavelength.
        """
        if spectral_units is None:
            return self.spectral_axis

        new_unit = validate_units(spectral_units)
        if new_unit == self.spectral_axis_unit:
            return self.spectral_axis

        if self.laser_wavelength is None:
            if Units.raman_shift in (new_unit, self.spectral_axis_unit):
                msg = (
                    f"Cannot convert the spectral axis of {self.title!r} to {new_unit}: the file "
                    "does not record the excitation wavelength."
                )
                raise ValueError(msg)
            return convert_spectral_units(self.spectral_axis, self.spectral_axis_unit, new_unit)

        return convert_spectral_units(
            self.spectral_axis,
            self.spectral_axis_unit,
            new_unit,
            laser_wavelength_nm=self.laser_wavelength,
        )

    def to_df(
        self,
        spectral_units: Units | Literal["nm", "cm-1", "eV", "raman_shift"] | None = None,
    ) -> pd.DataFrame:
        """Export the spectrum to a DataFrame.

        The spectral axis becomes the index, and the intensities a single column named after the
        title of the spectrum.

        Parameters
        ----------
        spectral_units : Units | {"nm", "cm-1", "eV", "raman_shift"} | None, optional
            Units in which the spectral axis will be exported, by default None (the units stored
            in the file).

        Returns
        -------
        pd.DataFrame
            A single-column DataFrame indexed by the spectral axis.

        Notes
        -----
        This layout is transposed with respect to :meth:`Mapping.to_df`, which puts one spectrum
        per *row* because every row is tied to a map coordinate. Spectra in a ``.mdt`` file are
        independent measurements, so a column per spectrum is the more natural layout.
        """
        axis_unit = (
            self.spectral_axis_unit if spectral_units is None else validate_units(spectral_units)
        )
        return pd.DataFrame(
            {self.title: self.data},
            index=pd.Index(self.get_spectral_axis(spectral_units), name=str(axis_unit)),
        )

    def to_csv(
        self,
        path: Path = Path(),
        filename: str = "",
        spectral_units: Units | Literal["nm", "cm-1", "eV", "raman_shift"] | None = None,
    ) -> Path:
        """Export the spectrum to a csv file.

        The first column holds the spectral axis in the selected units, and the second the
        intensities.

        Parameters
        ----------
        path : Path, optional
            Folder in which the file will be saved, by default Path().
        filename : str, optional
            Name to use for the file, by default "" (the sanitized title of the spectrum).
            Any extension will be replaced by ".csv".
        spectral_units : Units | {"nm", "cm-1", "eV", "raman_shift"} | None, optional
            Units in which the spectral axis will be exported, by default None.

        Returns
        -------
        Path
            The path of the written file.
        """
        stem = Path(filename).with_suffix("").as_posix() if filename else self.title
        file_path = path / (clean_filename(stem) + ".csv")

        path.mkdir(parents=True, exist_ok=True)
        self.to_df(spectral_units).to_csv(file_path, na_rep="NaN")
        return file_path


def _mdt_unit_name(unit_code: int) -> str:
    """Return a readable name for a raw NT-MDT unit code.

    Parameters
    ----------
    unit_code : int
        The raw code stored in the file.

    Returns
    -------
    str
        The name of the matching :class:`~nanofinderparser.units.MdtUnit` member, or the code
        itself as a string when it is not a known unit.
    """
    try:
        return MdtUnit(unit_code).name
    except ValueError:
        logger.warning("Unknown MDT unit code %d.", unit_code)
        return str(unit_code)


@dataclass(frozen=True, slots=True, eq=False)
class Image:
    """A 2-D scalar map, as stored in a NanoFinder ``.mdt`` file.

    NanoFinder uses these both for maps measured directly (for instance a PL intensity map) and
    for maps of parameters obtained by fitting each spectrum of a mapping, such as the position,
    intensity, or FWHM of a peak. Unlike :class:`Mapping`, an Image holds a single value per
    point rather than a whole spectrum.

    Attributes
    ----------
    title : str
        Name given to the map in NanoFinder, for example ``"G Peak position (Lorentz)"``.
    values : NDArray[np.float64]
        The map, of shape ``(y_size, x_size)``, in the units given by `value_unit`.
    x_axis, y_axis : AxisSpec
        Start position, step, and units of the two spatial axes.
    value_unit : str
        Units of the mapped values, for example ``"counts"`` or ``"raman_shift"``.
    measured_at : datetime | None
        Acquisition timestamp, or None when the file stores an invalid date.
    text_comment : str
        Free-text comment attached to the map, empty when absent.

    Properties
    ----------
    shape : tuple[int, int]
        Shape of the map, as ``(y_size, x_size)``.
    x_coords, y_coords : NDArray[np.float64]
        Physical coordinates along each axis.
    datetime : datetime | None
        Date and time of the measurement.
    date : date | None
        Date of the measurement.

    Methods
    -------
    to_df()
        Export the map to a DataFrame indexed by the y coordinates.
    to_csv(path=Path(), filename="")
        Export the map to a csv file.

    Notes
    -----
    Maps are stored as signed ``int16`` levels scaled to cover the range of the map, so their
    resolution is limited to roughly one 65536th of that range. This is finer than one count for
    most maps, but not for those covering a wide range of values: a map running from 1e5 to 2e5
    counts only resolves steps of about 1.7 counts, and the original values cannot be recovered
    beyond that.

    The unit codes written by NanoFinder for the spatial axes are not always self-consistent:
    some files declare ``meter`` for step sizes that are clearly in micrometers. The codes are
    reported as they are stored, without reinterpretation.
    """

    title: str
    values: NDArray[np.float64]
    x_axis: AxisSpec
    y_axis: AxisSpec
    value_unit: str
    measured_at: datetime | None
    text_comment: str = ""

    @classmethod
    def from_mdt_frame(cls, frame: MdtImageFrame) -> Self:
        """Build an Image from a raw frame read out of a ``.mdt`` file.

        Parameters
        ----------
        frame : MdtImageFrame
            The decoded frame, as returned by
            :func:`~nanofinderparser.parsers.read_mdt_frames`.

        Returns
        -------
        Image
            The corresponding map.
        """
        return cls(
            title=frame.title,
            values=frame.values,
            x_axis=AxisSpec(
                frame.x_scale.offset, frame.x_scale.step, _mdt_unit_name(frame.x_scale.unit_code)
            ),
            y_axis=AxisSpec(
                frame.y_scale.offset, frame.y_scale.step, _mdt_unit_name(frame.y_scale.unit_code)
            ),
            value_unit=_mdt_unit_name(frame.z_scale.unit_code),
            measured_at=frame.measured_at,
            text_comment=frame.text_comment,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the map, as ``(y_size, x_size)``."""
        y_size, x_size = self.values.shape
        return int(y_size), int(x_size)

    @property
    def x_coords(self) -> NDArray[np.float64]:
        """Physical coordinates along the x (fast) axis."""
        coords = self.x_axis.start + np.arange(self.shape[1]) * self.x_axis.step
        return coords.astype(np.float64)

    @property
    def y_coords(self) -> NDArray[np.float64]:
        """Physical coordinates along the y (slow) axis."""
        coords = self.y_axis.start + np.arange(self.shape[0]) * self.y_axis.step
        return coords.astype(np.float64)

    @property
    def datetime(self) -> datetime | None:
        """Date and time of the measurement."""
        return self.measured_at

    @property
    def date(self) -> date | None:
        """Date of the measurement."""
        return None if self.measured_at is None else self.measured_at.date()

    def to_df(self) -> pd.DataFrame:
        """Export the map to a DataFrame.

        The y coordinates become the index and the x coordinates the columns, so the DataFrame
        has the same layout as the map itself.

        Returns
        -------
        pd.DataFrame
            The map, indexed by physical coordinates.
        """
        return pd.DataFrame(
            self.values,
            index=pd.Index(self.y_coords, name=f"y ({self.y_axis.units})"),
            columns=pd.Index(self.x_coords, name=f"x ({self.x_axis.units})"),
        )

    def to_csv(self, path: Path = Path(), filename: str = "") -> Path:
        """Export the map to a csv file.

        Parameters
        ----------
        path : Path, optional
            Folder in which the file will be saved, by default Path().
        filename : str, optional
            Name to use for the file, by default "" (the sanitized title of the map).
            Any extension will be replaced by ".csv".

        Returns
        -------
        Path
            The path of the written file.
        """
        stem = Path(filename).with_suffix("").as_posix() if filename else self.title
        file_path = path / (clean_filename(stem) + ".csv")

        path.mkdir(parents=True, exist_ok=True)
        self.to_df().to_csv(file_path, na_rep="NaN")
        return file_path


class Titled(Protocol):
    """Protocol for the items held by a :class:`TitledSequence`."""

    @property
    def title(self) -> str:
        """Name of the item."""
        ...


class TitledSequence[ItemT: Titled](Sequence[ItemT]):
    """Read-only sequence of titled items, that also allows lookup by title.

    Attributes
    ----------
    source : Path | None
        Path of the file the items were read from, when known.
    """

    def __init__(self, items: Sequence[ItemT], source: Path | None = None) -> None:
        """Initialize the collection.

        Parameters
        ----------
        items : Sequence[ItemT]
            The items to hold, in file order.
        source : Path | None, optional
            Path of the file the items were read from, by default None.
        """
        self._items: list[ItemT] = list(items)
        self.source = source

    def __len__(self) -> int:
        """Items in the collection."""
        return len(self._items)

    def __iter__(self) -> Iterator[ItemT]:
        """Iterate over the items in file order."""
        return iter(self._items)

    @overload
    def __getitem__(self, key: int | str) -> ItemT: ...
    @overload
    def __getitem__(self, key: slice) -> Self: ...
    def __getitem__(self, key: int | str | slice) -> "ItemT | Self":
        """Get an item by position or title, or a sub-collection by slice.

        Parameters
        ----------
        key : int | str | slice
            The position of the item, its title, or a slice of positions.

        Returns
        -------
        ItemT | Self
            The requested item, or a collection of the same type when `key` is a slice.

        Raises
        ------
        KeyError
            If `key` is a title that no item in the collection has.
        """
        if isinstance(key, str):
            for item in self._items:
                if item.title == key:
                    return item
            msg = f"No item titled {key!r}. Available titles: {self.titles}"
            raise KeyError(msg)
        if isinstance(key, slice):
            return type(self)(self._items[key], source=self.source)
        return self._items[key]

    def __repr__(self) -> str:
        """Concise representation listing the titles of the items."""
        return f"{type(self).__name__}({self.titles!r})"

    @property
    def titles(self) -> list[str]:
        """Titles of the items, in file order."""
        return [item.title for item in self._items]

    def unique_titles(self) -> list[str]:
        """Return the titles, disambiguating duplicates with a numeric suffix.

        Returns
        -------
        list[str]
            The titles, where the n-th repetition of a title is suffixed with ``_n``.
        """
        seen: dict[str, int] = {}
        unique: list[str] = []
        for title in self.titles:
            count = seen.get(title, 0)
            seen[title] = count + 1
            unique.append(title if count == 0 else f"{title}_{count + 1}")
        return unique


class Spectra(TitledSequence[Spectrum]):
    """The individual spectra stored in a NanoFinder ``.mdt`` file.

    Behaves as a read-only sequence of :class:`Spectrum` objects, and additionally allows lookup
    by title. It is recommended to create instances using
    :func:`~nanofinderparser.load.load_mdt` rather than instantiating this class directly.

    Methods
    -------
    to_df(spectral_units=None)
        Export every spectrum to a single DataFrame, one column per spectrum.
    to_csv(path=Path(), filename="", spectral_units=None, combined=False)
        Export the spectra to csv file(s).

    Examples
    --------
    >>> from pathlib import Path
    >>> spectra = load_mdt(Path("path/to/your/file.mdt"))  # doctest: +SKIP
    >>> spectra.titles  # doctest: +SKIP
    ['Spectrum_1', 'Spectrum_2']
    >>> spectra["Spectrum_1"].laser_wavelength  # doctest: +SKIP
    532.0
    """

    def to_df(
        self,
        spectral_units: Units | Literal["nm", "cm-1", "eV", "raman_shift"] | None = None,
    ) -> pd.DataFrame:
        """Export every spectrum to a single DataFrame, one column per spectrum.

        Parameters
        ----------
        spectral_units : Units | {"nm", "cm-1", "eV", "raman_shift"} | None, optional
            Units in which the spectral axis will be exported, by default None.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by the spectral axis, with one column per spectrum.

        Notes
        -----
        Spectra in the same file may have been recorded with different grating positions, and
        therefore over different spectral axes. In that case the columns are aligned on the union
        of the axes, which leaves missing values, and a warning is emitted; exporting each
        spectrum separately is usually more appropriate.
        """
        if not self._items:
            return pd.DataFrame()

        frames = [spectrum.to_df(spectral_units) for spectrum in self._items]
        for frame, title in zip(frames, self.unique_titles(), strict=True):
            frame.columns = pd.Index([title])

        reference = frames[0].index
        if any(not frame.index.equals(reference) for frame in frames[1:]):
            logger.warning(
                "The spectra of %s do not share a common spectral axis; the combined DataFrame "
                "will contain missing values.",
                self.source or "this collection",
            )

        return pd.concat(frames, axis=1)

    def to_csv(
        self,
        path: Path = Path(),
        filename: str = "",
        spectral_units: Units | Literal["nm", "cm-1", "eV", "raman_shift"] | None = None,
        combined: bool = False,
    ) -> list[Path]:
        """Export the spectra to csv file(s).

        Parameters
        ----------
        path : Path, optional
            Folder in which the file(s) will be saved, by default Path().
        filename : str, optional
            Base name for the file(s), by default "". When exporting separately, it is used as a
            prefix for the title of each spectrum; when exporting combined, it is the name of the
            single file. Any extension will be replaced by ".csv".
        spectral_units : Units | {"nm", "cm-1", "eV", "raman_shift"} | None, optional
            Units in which the spectral axis will be exported, by default None.
        combined : bool, optional
            If True, write all the spectra to a single file. If False (the default), write one
            file per spectrum, which is safer when the spectra do not share a spectral axis.

        Returns
        -------
        list[Path]
            The paths of the written files.
        """
        stem = Path(filename).with_suffix("").as_posix() if filename else ""
        path.mkdir(parents=True, exist_ok=True)

        if combined:
            file_path = path / (clean_filename(stem) + ".csv")
            self.to_df(spectral_units).to_csv(file_path, na_rep="NaN")
            return [file_path]

        written: list[Path] = []
        for spectrum, title in zip(self._items, self.unique_titles(), strict=True):
            name = f"{stem}_{title}" if stem else title
            written.append(spectrum.to_csv(path, filename=name, spectral_units=spectral_units))
        return written


class Images(TitledSequence[Image]):
    """The 2-D scalar maps stored in a NanoFinder ``.mdt`` file.

    Behaves as a read-only sequence of :class:`Image` objects, and additionally allows lookup by
    title. It is recommended to create instances using
    :func:`~nanofinderparser.load.load_mdt_images` rather than instantiating this class directly.

    Methods
    -------
    to_csv(path=Path(), filename="")
        Export the maps to one csv file each.

    Examples
    --------
    >>> from pathlib import Path
    >>> images = load_mdt_images(Path("path/to/your/file.mdt"))  # doctest: +SKIP
    >>> images[0].shape  # doctest: +SKIP
    (30, 30)
    """

    def to_csv(self, path: Path = Path(), filename: str = "") -> list[Path]:
        """Export the maps to one csv file each.

        Maps in the same file generally have different value ranges and units, so they are always
        written separately.

        Parameters
        ----------
        path : Path, optional
            Folder in which the files will be saved, by default Path().
        filename : str, optional
            Prefix for the file names, by default "" (only the title of each map is used).
            Any extension will be replaced by ".csv".

        Returns
        -------
        list[Path]
            The paths of the written files.
        """
        stem = Path(filename).with_suffix("").as_posix() if filename else ""
        path.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for image, title in zip(self._items, self.unique_titles(), strict=True):
            name = f"{stem}_{title}" if stem else title
            written.append(image.to_csv(path, filename=name))
        return written
