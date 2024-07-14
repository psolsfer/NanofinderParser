"""Models to parse nanofinder files."""

from datetime import date, datetime, time
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_core.core_schema import ValidationInfo

# from pydataset.spectraset import SpectraSet
from nanofinderparser.map import _nanofinder_mapcoords
from nanofinderparser.units import CONVERSION_UNITS, convert_spectral_units

# Type for generic numpy arrays
NPDtype_co = TypeVar("NPDtype_co", bound=np.generic, covariant=True)


class VendorVersion(BaseModel):
    Vendor: str  # vendor: str = Field(alias="Vendor") # NOTE aliases can also be used
    Version: str


class FrameHeader(VendorVersion, BaseModel):
    Date: date
    Time: time
    Information: str
    SystemName: str
    PositioningSysName: str
    DetectionSysName: str
    ScannedDataName: str  # IMPORTANT???

    @field_validator("Date", mode="before")
    def parse_date(cls, value: str, info: ValidationInfo) -> str:
        """To properly parse the date."""
        return value.replace("/", "-")

    @property
    def datetime(self) -> datetime:
        """Date and time of the measurement."""
        # ??? Change the format, as in datetime.strptime(date + " " + time, "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)?
        return datetime.combine(self.Date, self.Time)


class FrameOptions(VendorVersion, BaseModel):
    OmuLaserWLnm: float  # IMPORTANT Wavelength of the laser in nm
    OmuCurPower: float
    OmuGratingGroove: str
    OmuCentralWaveLengthNM: float  # IMPORTANT Wavelength center for the spectrum
    OmuPinHoleSize: float


class Axis(BaseModel):
    AxisIsInUse: int
    AxisIsInversed: bool
    AxisName: str
    AxisUnitName: str  # IMPORTANT Units of the axis
    AxisCountStep: int
    AxisScaleFloat: float

    @property
    def step_size(self) -> float:
        """Step size in AxisName units."""
        return self.AxisCountStep * self.AxisScaleFloat

    @property
    def step_units(self) -> str:
        """Units of the step."""
        return self.AxisUnitName


class StageAxesDimentions(BaseModel):
    AxisX: Axis
    AxisY: Axis
    AxisZ: Axis

    @property
    def step_size(self) -> tuple[float, float, float]:
        """Size of the map steps in the (x,y,z) axes."""
        return (self.AxisX.step_size, self.AxisY.step_size, self.AxisZ.step_size)

    @property
    def step_units(self) -> tuple[str, str, str]:
        """Units of the map steps in the (x,y,z) axes."""
        return (self.AxisX.step_units, self.AxisY.step_units, self.AxisZ.step_units)


class Stage3DParameters(VendorVersion, BaseModel):
    AxisSizeX: int  # IMPORTANT Number of steps of mapping
    AxisSizeY: int  # IMPORTANT Number of steps of mapping
    AxisSizeZ: int
    StageAxesDimentions: StageAxesDimentions

    @property
    def map_steps(self) -> tuple[int, int, int]:
        """Number of steps of the map in the (x,y,z) axes."""
        return (self.AxisSizeX, self.AxisSizeY, self.AxisSizeY)


class Channel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # To allow having numpy arrays

    DeviceGuid: str
    DeviceName: str
    DataChannelName: str  # For example "Photons"
    DataChannelUnit: str  # For example "Counts"
    ChannelSize: int  # IMPORTANT Number of data points composing each spectrum
    ChannelAxisName: str  # For example "Wavelength"
    ChannelAxisUnit: Literal["nm", "cm-1", "eV"]  # For example "nm"
    ChannelAxisLaserWl: float  # IMPORTANT Wavelength excitation
    ChannelAxisArray: NDArray[
        NPDtype_co
    ]  # IMPORTANT # Spectral axis, with units given by ChannelAxisUnit

    @field_validator("ChannelAxisArray", mode="before")
    def parse_chanelaxisarray(cls, value: str, info: ValidationInfo) -> NDArray[NPDtype_co]:
        """To properly parse the ChannelAxisArray."""
        return np.fromstring(value, sep=" ")


class ScannedFrameParameters(VendorVersion, BaseModel):
    ScanRepeatNumber: int
    FrameHeader: FrameHeader
    FrameOptions: FrameOptions
    Stage3DParameters: Stage3DParameters
    Channel: Channel


class Mapping(VendorVersion, BaseModel):
    """Data obtained from a mapping .smd file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # To allow having numpy arrays

    ScannedFrameParameters: ScannedFrameParameters
    Data: NDArray[NPDtype_co]  # Includes the data for the spectra

    @property
    def spectral_axis(self) -> NDArray[NPDtype_co]:
        """Array containing the spectral axis."""
        return self.ScannedFrameParameters.Channel.ChannelAxisArray

    @property
    def spectral_axis_len(self) -> int:
        """Number of data points of each spectrum."""
        return self.ScannedFrameParameters.Channel.ChannelSize

    @property
    def laser_wavelength(self) -> float:
        """Wavelength of the laser in nm."""
        return self.ScannedFrameParameters.FrameOptions.OmuLaserWLnm

    @property
    def datetime(self) -> datetime:
        """Date and time of the measurement."""
        return self.ScannedFrameParameters.FrameHeader.datetime

    @property
    def date(self) -> date:
        """Date of the measurement."""
        return self.ScannedFrameParameters.FrameHeader.Date

    @property
    def step_size(self) -> tuple[float, float, float]:
        """Size of the map steps in the (x,y,z) axes."""
        return self.ScannedFrameParameters.Stage3DParameters.StageAxesDimentions.step_size

    @property
    def step_units(self) -> tuple[str, str, str]:
        """Units of the map steps in the (x,y,z) axes."""
        return self.ScannedFrameParameters.Stage3DParameters.StageAxesDimentions.step_units

    @property
    def map_steps(self) -> tuple[int, int, int]:
        """Number of steps of the map in the (x,y,z) axes."""
        return self.ScannedFrameParameters.Stage3DParameters.map_steps

    @property
    def map_size(self) -> tuple[float, float, float]:
        """Size of the map in the (x,y,z) axes, with the corresponding units for each axis."""
        return tuple(x * y for x, y in zip(self.step_size, self.map_steps, strict=True))

    @property
    def _data_to_map(self) -> NDArray[NPDtype_co]:
        """Reshapes the data as the mapping: (x, y, spectrum)."""
        return self.Data.reshape((self.map_steps[0], self.map_steps[1], self.spectral_axis_len))

    @property
    def _channel_axis_unit(self) -> Literal["nm", "cm-1", "eV"]:
        """Units of the spectral axis."""
        return self.ScannedFrameParameters.Channel.ChannelAxisUnit

    def export_to_csv(
        self,
        path: Path = Path(),
        filename: str = "",
        spectral_units: CONVERSION_UNITS | None = None,
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
            Suffix to use for the name of the files, by default ""
        spectral_units : {"nm", "cm-1", "eV", "raman_shift"}, optional
            Units in which the spectral axis will be exported, by default None
        """
        data, mapcoords = self.export_to_df(spectral_units)

        if not filename:
            map_file_path = path / "data.csv"
            coord_file_path = path / "mapcoords.csv"
        else:
            map_file_path = path / (filename + "_data.csv")
            coord_file_path = path / (filename + "mapcoords.csv")

        data.to_csv(map_file_path, na_rep="NaN", index=False)
        mapcoords.to_csv(coord_file_path, na_rep="NaN", index=False)

    def export_to_df(
        self,
        spectral_units: CONVERSION_UNITS | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export the data and mapcoords to DataFrames.

        It exports the data of the spectra and the mapping coordinates as pandas DataFrames.
        For the data, the header corresponds to the spectral axis in the selected units, and each
        row corresponds to a single spectrum.
        Each row of the mapping coordinates file provides the coordinates of the spectrum in the
        same row.

        Parameters
        ----------
        spectral_units : {"nm", "cm-1", "eV", "raman_shift"}, optional
            Units in which the spectral axis will be exported, by default None

        Returns
        -------
        The data and mapping coordinates as DataFrames.
        """
        spectral_axis = self._to_spectral_units(spectral_units)

        mapcoords = _nanofinder_mapcoords(
            self.map_steps[0], self.map_steps[1]
        )  # FIXME Doesn't work for Z-axis

        # Reordering the rows
        # NOTE: this is not essential, only done to conincide with NanoFinder's convention of 'y'
        # starting from the bottom side of the mapping area
        mapcoords = mapcoords.sort_values(by=["y", "x"], ascending=[False, True])

        data = pd.DataFrame(self.Data, columns=spectral_axis).reindex(mapcoords.index)

        return data, mapcoords

    # def export_to_dataset(self, dataset_kind=SpectraSet) -> SpectraSet:
    #     # TODO # ISSUE #50 Export the Mapping to a dataset
    #     # Accepts the kind of dataset as an argument, and returns that same kind of dataset
    #     pass

    def _to_spectral_units(self, new_unit: CONVERSION_UNITS | None) -> NDArray[np.float64]:
        """Convert the spectral units to the given ones.

        Parameters
        ----------
        new_unit : {"nm", "cm-1", "eV", "raman_shift"}
            The units to convert the spectral axis.

        Returns
        -------
        The converted spectral axis
        """
        if new_unit is None or self._channel_axis_unit == new_unit:
            return self.spectral_axis

        return convert_spectral_units(
            self.spectral_axis,
            self._channel_axis_unit,
            new_unit,
            laser_wavelength_nm=self.laser_wavelength,
        ).magnitude

    @field_validator("Data", mode="before")
    def parse_data(cls, value: str, info: ValidationInfo) -> NDArray[NPDtype_co]:
        """To properly parse the Data."""
        return np.asarray(value)  # TODO Use dtype = int??? Or does some kind of data be float???

    def model_post_init(self, __context: None) -> None:
        """Reshape the Data into a row per spectrum."""
        self.Data = self.Data.reshape(-1, self.spectral_axis_len)
