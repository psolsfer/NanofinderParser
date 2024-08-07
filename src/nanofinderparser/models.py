"""Models to parse nanofinder files."""

from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

# from pydataset.spectraset import SpectraSet
from nanofinderparser.map import _nanofinder_mapcoords
from nanofinderparser.units import CONVERSION_UNITS, convert_spectral_units

# Type for generic numpy arrays
NPDtype_co = TypeVar("NPDtype_co", bound=np.generic, covariant=True)


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
    def parse_date(cls, value: str, info: ValidationInfo) -> date:
        """To properly parse the date."""
        return datetime.strptime(value, "%Y/%m/%d").date()

    @field_validator("time_model", mode="before")
    @classmethod
    def parse_time(cls, value: str, info: ValidationInfo) -> time:
        """To properly parse the time."""
        return datetime.strptime(value, "%H:%M:%S").time()

    @property
    def datetime(self) -> datetime:
        """Date and time of the measurement."""
        # ??? Change the format, as in datetime.strptime(date + " " + time, "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)?
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
        Indicator if the axis is in use.
    is_inversed : bool
        Indicator if the axis is inversed.
    name : str
        The name of the axis.
    unit_name : str
        The unit name for the axis.
    count_step : int
        The count step for the axis.
    scale_float : float
        The scale factor for the axis.

    Methods
    -------
    step_size : float
        Property that calculates the step size.
    step_units : str
        Property that returns the step units.
    """

    is_in_use: int = Field(alias="AxisIsInUse")
    is_inversed: bool = Field(alias="AxisIsInversed")
    name: str = Field(alias="AxisName")
    unit_name: str = Field(alias="AxisUnitName")  # IMPORTANT Units of the axis
    count_step: int = Field(alias="AxisCountStep")
    scale_float: float = Field(alias="AxisScaleFloat")

    @property
    def step_size(self) -> float:
        """Step size in AxisName units."""
        return self.count_step * self.scale_float

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
        return (self.x.step_size, self.y.step_size, self.x.step_size)

    @property
    def step_units(self) -> tuple[str, str, str]:
        """Units of the map steps in the (x,y,z) axes."""
        return (self.x.step_units, self.y.step_units, self.z.step_units)


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


class ChannelInfo(BaseModel):
    """Model for detailed channel information.

    This class represents additional information about the channel,
    including temperature, exposure time, and acquisition mode.

    Attributes
    ----------
    temperature : float | None
        The temperature of the CCD in degrees Celsius.
    exposure_time : float | None
        The exposure time for each acquisition in seconds.
    cycle_time : float | None
        The total cycle time for each acquisition in seconds.
    acquisition_mode : str | None
        The mode of acquisition (e.g., "accumulate", "single").
    accumulation_number : int | None
        The number of accumulations for accumulate mode.

    Notes
    -----
    All attributes are optional (can be None) to accommodate varying levels of available information
    from different data sources.
    """

    temperature: float | None = None
    exposure_time: float | None = None
    cycle_time: float | None = None
    acquisition_mode: str | None = None
    accumulation_number: int | None = None


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
    channel_axis_array : List[float]
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
    channel_axis_array: NDArray = Field(
        alias="ChannelAxisArray"
    )  # IMPORTANT # Spectral axis, with units given by ChannelAxisUnit

    channel_info: ChannelInfo = Field(alias="ChannelInfo")

    @field_validator("channel_axis_array", mode="before")
    @classmethod
    def parse_chanelaxisarray(cls, value: str, info: ValidationInfo) -> NDArray[NPDtype_co]:
        """Properly parse the ChannelAxisArray."""
        return np.fromstring(value, sep=" ")

    @field_validator("channel_info", mode="before")
    @classmethod
    def parse_channel_info(cls, value: dict[str, str], info: ValidationInfo) -> ChannelInfo:
        info_dict = {}
        for v in value.values():
            if "Temperature" in v:
                info_dict["temperature"] = float(v.split("=")[1].strip())
            elif "Exposure time" in v:
                info_dict["exposure_time"] = float(v.split("=")[1].strip().split()[0])
                info_dict["cycle_time"] = float(v.split("=")[2].strip())
            elif "Acquisition mode" in v:
                mode_info = v.split(":")[1].strip().split(".")
                info_dict["acquisition_mode"] = mode_info[0].strip().lower()
                if info_dict["acquisition_mode"] in ["accomulate", "accumulate"]:
                    info_dict["accumulation_number"] = int(mode_info[1].split("=")[1].strip())
        return ChannelInfo(**info_dict)


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
    """

    scan_repeat_number: int = Field(alias="ScanRepeatNumber")
    frame_header: FrameHeader = Field(alias="FrameHeader")
    frame_options: FrameOptions = Field(alias="FrameOptions")
    stage_3d_parameters: Stage3DParameters = Field(alias="Stage3DParameters")
    data_calibration: DataCalibration = Field(alias="DataCalibration")


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
        The actual mapping data.

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
    step_size : tuple[float, float, float]
        Size of the map steps in the (x,y,z) axes.
    step_units : tuple[str, str, str]
        Units of the map steps in the (x,y,z) axes.
    map_steps : tuple[int, int, int]
        Number of steps of the map in the (x,y,z) axes.
    map_size : tuple[float, float, float]
        Size of the map in the (x,y,z) axes, with the corresponding units for each axis.

    Methods
    -------
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
    export_to_csv(path: Path = Path(), filename: str = "", spectral_units: CONVERSION_UNITS | None = None, save_mapcoords: bool = False, channel: int = 0)
        Export the data to csv files.
    export_to_df(spectral_units: CONVERSION_UNITS | None = None, channel: int = 0)
        Export the data and mapcoords to DataFrames.

    Notes
    -----
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
        """The actual mapping data."""
        return self._data

    @data.setter
    def data(self, value: list[float]) -> None:
        value_array = np.asarray(value)  # dtype will be inferred

        # Reshape the Data into a row per spectrum.
        # FIXME This won't handle the case when there are more than one channel.
        # Need a SMD file with several channels to inspect it and implement handling multichannels.
        channel = 0
        self._data = value_array.reshape(-1, self.get_spectral_axis_len(channel=0))

    def get_spectral_axis(self, channel: int = 0) -> NDArray[NPDtype_co]:
        """Get the spectral axis for the given channel.

        Parameters
        ----------
        channel : int, optional
            The channel index, by default 0

        Returns
        -------
        NDArray[NPDtype_co]
            Array containing the spectral axis for the given channel.
        """
        channel_obj = self.scanned_frame_parameters.data_calibration.channels[channel]
        return channel_obj.channel_axis_array

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
    def map_size(self) -> tuple[float, float, float]:
        """Size of the map in the (x,y,z) axes, with the corresponding units for each axis."""
        return tuple(x * (y - 1) for x, y in zip(self.step_size, self.map_steps, strict=True))

    def _get_data_to_map(self, channel: int = 0) -> NDArray[NPDtype_co]:
        """Reshapes the data as the mapping: (x, y, spectrum)."""
        return self.data.reshape(
            (self.map_steps[0], self.map_steps[1], self.get_spectral_axis_len(channel))
        )

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

    def export_to_csv(
        self,
        path: Path = Path(),
        filename: str = "",
        spectral_units: CONVERSION_UNITS | None = None,
        save_mapcoords: bool = False,
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
        spectral_units : {"nm", "cm-1", "eV", "raman_shift"}, optional
            Units in which the spectral axis will be exported, by default None
        save_mapcoords : bool, optional
            Whether to save the file with the mapcoords. By default False.
        channel : int, optional
            The channel index to export, by default 0
        """
        data, mapcoords = self.export_to_df(spectral_units, channel=channel)

        if not filename:
            map_file_path = path / "data.csv"
            coord_file_path = path / "mapcoords.csv"
        else:
            filename = Path(filename).with_suffix("").as_posix()
            if save_mapcoords:
                map_file_path = path / (filename + "_data.csv")
            else:
                map_file_path = path / (filename + ".csv")
            coord_file_path = path / (filename + "_mapcoords.csv")

        data.to_csv(map_file_path, na_rep="NaN", index=False)
        if save_mapcoords:
            mapcoords.to_csv(coord_file_path, na_rep="NaN", index=False)

    def export_to_df(
        self,
        spectral_units: CONVERSION_UNITS | None = None,
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
        spectral_units : {"nm", "cm-1", "eV", "raman_shift"}, optional
            Units in which the spectral axis will be exported, by default None
        channel : int, optional
            The channel index to export, by default 0

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            The data and mapping coordinates as DataFrames.
        """
        spectral_axis = self._to_spectral_units(spectral_units, channel=channel)

        mapcoords = _nanofinder_mapcoords(
            self.map_steps[0], self.map_steps[1]
        )  # FIXME Doesn't work for Z-axis

        # Reordering the rows
        # NOTE: this is not essential, only done to coincide with NanoFinder's convention of 'y'
        # starting from the bottom side of the mapping area
        mapcoords = mapcoords.sort_values(by=["y", "x"], ascending=[False, True])

        data = pd.DataFrame(self.data, columns=spectral_axis).reindex(mapcoords.index)

        return data, mapcoords

    # def export_to_dataset(self, dataset_kind=SpectraSet) -> SpectraSet:
    #     # TODO # ISSUE #50 Export the Mapping to a dataset
    #     # Accepts the kind of dataset as an argument, and returns that same kind of dataset
    #     pass

    def _to_spectral_units(
        self, new_unit: CONVERSION_UNITS | None, channel: int = 0
    ) -> NDArray[np.float64]:
        """Convert the spectral units to the given ones.

        Parameters
        ----------
        new_unit : {"nm", "cm-1", "eV", "raman_shift"}
            The units to convert the spectral axis.
        channel : int, optional
            The channel index, by default 0

        Returns
        -------
        NDArray[np.float64]
            The converted spectral axis
        """
        if new_unit is None or self._get_channel_axis_unit(channel) == new_unit:
            return self.get_spectral_axis(channel)

        return convert_spectral_units(
            self.get_spectral_axis(channel),
            self._get_channel_axis_unit(channel),
            new_unit,
            laser_wavelength_nm=self.laser_wavelength,
        )
