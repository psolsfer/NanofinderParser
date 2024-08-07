from datetime import date, datetime, time

import numpy as np
import pytest
from nanofinderparser.models import (
    Axis,
    Channel,
    DataCalibration,
    FrameHeader,
    FrameOptions,
    Mapping,
    ScannedFrameParameters,
    Stage3DParameters,
    StageAxesDimensions,
    VendorVersion,
)


def test_vendor_version():
    vv = VendorVersion(Vendor="TestVendor", Version="1.0")
    assert vv.vendor == "TestVendor"
    assert vv.version == "1.0"


def test_frame_header():
    fh = FrameHeader(
        Vendor="TestVendor",
        Version="1.0",
        Date="2023/01/01",
        Time="12:00:00",
        Information="Test",
        SystemName="TestSystem",
        PositioningSysName="TestPos",
        DetectionSysName="TestDet",
        ScannedDataName="TestData",
    )
    assert fh.datetime == datetime(2023, 1, 1, 12, 0, 0)


def test_frame_options():
    fo = FrameOptions(
        Vendor="TestVendor",
        Version="1.0",
        OmuLaserWLnm=532.0,
        OmuCurPower=10.0,
        OmuGratingGroove="1800",
        OmuCentralWaveLengthNM=550.0,
        OmuPinHoleSize=100.0,
    )
    assert fo.laser_wavelength_nm == 532.0


def test_axis():
    ax = Axis(
        AxisIsInUse=1,
        AxisIsInversed=False,
        AxisName="X",
        AxisUnitName="um",
        AxisCountStep=100,
        AxisScaleFloat=0.1,
    )
    assert ax.step_size == 10.0
    assert ax.step_units == "um"


def test_stage_axes_dimensions():
    sad = StageAxesDimensions(
        AxisX=Axis(
            AxisIsInUse=1,
            AxisIsInversed=False,
            AxisName="X",
            AxisUnitName="um",
            AxisCountStep=100,
            AxisScaleFloat=0.1,
        ),
        AxisY=Axis(
            AxisIsInUse=1,
            AxisIsInversed=False,
            AxisName="Y",
            AxisUnitName="um",
            AxisCountStep=100,
            AxisScaleFloat=0.1,
        ),
        AxisZ=Axis(
            AxisIsInUse=1,
            AxisIsInversed=False,
            AxisName="Z",
            AxisUnitName="um",
            AxisCountStep=100,
            AxisScaleFloat=0.1,
        ),
    )
    assert sad.step_size == (10.0, 10.0, 10.0)
    assert sad.step_units == ("um", "um", "um")


def test_stage_3d_parameters():
    s3dp = Stage3DParameters(
        Vendor="TestVendor",
        Version="1.0",
        AxisSizeX=10,
        AxisSizeY=10,
        AxisSizeZ=1,
        StageAxesDimentions=StageAxesDimensions(
            AxisX=Axis(
                AxisIsInUse=1,
                AxisIsInversed=False,
                AxisName="X",
                AxisUnitName="um",
                AxisCountStep=100,
                AxisScaleFloat=0.1,
            ),
            AxisY=Axis(
                AxisIsInUse=1,
                AxisIsInversed=False,
                AxisName="Y",
                AxisUnitName="um",
                AxisCountStep=100,
                AxisScaleFloat=0.1,
            ),
            AxisZ=Axis(
                AxisIsInUse=1,
                AxisIsInversed=False,
                AxisName="Z",
                AxisUnitName="um",
                AxisCountStep=100,
                AxisScaleFloat=0.1,
            ),
        ),
    )
    assert s3dp.map_steps == (10, 10, 1)


def test_channel():
    ch = Channel(
        DeviceGuid="TestGuid",
        DeviceName="TestDevice",
        DataChannelName="TestChannel",
        DataChannelUnit="Counts",
        ChannelSize=1000,
        ChannelAxisName="Wavelength",
        ChannelAxisUnit="nm",
        ChannelAxisLaserWl=532.0,
        ChannelAxisArray="0 1 2 3 4",
        ChannelInfo={
            "Temperature": "Temperature = 25.0",
            "Exposure time": "Exposure time = 1.0, Cycle time = 1.1",
        },
    )
    assert ch.channel_axis_unit == "nm"
    assert np.array_equal(ch.channel_axis_array, np.array([0, 1, 2, 3, 4]))
    assert ch.channel_info.temperature == 25.0
    assert ch.channel_info.exposure_time == 1.0


def test_data_calibration():
    dc = DataCalibration(
        Vendor="TestVendor",
        Version="1.0",
        Channels=[
            Channel(
                DeviceGuid="TestGuid",
                DeviceName="TestDevice",
                DataChannelName="TestChannel",
                DataChannelUnit="Counts",
                ChannelSize=1000,
                ChannelAxisName="Wavelength",
                ChannelAxisUnit="nm",
                ChannelAxisLaserWl=532.0,
                ChannelAxisArray="0 1 2 3 4",
                ChannelInfo={
                    "Temperature": "Temperature = 25.0",
                    "Exposure time": "Exposure time = 1.0, Cycle time = 1.1",
                },
            )
        ],
    )
    assert len(dc.channels) == 1


def test_scanned_frame_parameters():
    sfp = ScannedFrameParameters(
        Vendor="TestVendor",
        Version="1.0",
        ScanRepeatNumber=1,
        FrameHeader=FrameHeader(
            Vendor="TestVendor",
            Version="1.0",
            Date="2023/01/01",
            Time="12:00:00",
            Information="Test",
            SystemName="TestSystem",
            PositioningSysName="TestPos",
            DetectionSysName="TestDet",
            ScannedDataName="TestData",
        ),
        FrameOptions=FrameOptions(
            Vendor="TestVendor",
            Version="1.0",
            OmuLaserWLnm=532.0,
            OmuCurPower=10.0,
            OmuGratingGroove="1800",
            OmuCentralWaveLengthNM=550.0,
            OmuPinHoleSize=100.0,
        ),
        Stage3DParameters=Stage3DParameters(
            Vendor="TestVendor",
            Version="1.0",
            AxisSizeX=10,
            AxisSizeY=10,
            AxisSizeZ=1,
            StageAxesDimentions=StageAxesDimensions(
                AxisX=Axis(
                    AxisIsInUse=1,
                    AxisIsInversed=False,
                    AxisName="X",
                    AxisUnitName="um",
                    AxisCountStep=100,
                    AxisScaleFloat=0.1,
                ),
                AxisY=Axis(
                    AxisIsInUse=1,
                    AxisIsInversed=False,
                    AxisName="Y",
                    AxisUnitName="um",
                    AxisCountStep=100,
                    AxisScaleFloat=0.1,
                ),
                AxisZ=Axis(
                    AxisIsInUse=1,
                    AxisIsInversed=False,
                    AxisName="Z",
                    AxisUnitName="um",
                    AxisCountStep=100,
                    AxisScaleFloat=0.1,
                ),
            ),
        ),
        DataCalibration=DataCalibration(
            Vendor="TestVendor",
            Version="1.0",
            Channels=[
                Channel(
                    DeviceGuid="TestGuid",
                    DeviceName="TestDevice",
                    DataChannelName="TestChannel",
                    DataChannelUnit="Counts",
                    ChannelSize=1000,
                    ChannelAxisName="Wavelength",
                    ChannelAxisUnit="nm",
                    ChannelAxisLaserWl=532.0,
                    ChannelAxisArray="0 1 2 3 4",
                    ChannelInfo={
                        "Temperature": "Temperature = 25.0",
                        "Exposure time": "Exposure time = 1.0, Cycle time = 1.1",
                    },
                )
            ],
        ),
    )
    assert sfp.scan_repeat_number == 1


def test_mapping(sample_mapping_data):
    assert sample_mapping_data.data.shape == (100, 1000)
    assert sample_mapping_data.step_size == (0.1, 0.1, 0.1)
    assert sample_mapping_data.step_units == ("um", "um", "um")
    assert sample_mapping_data.map_steps == (10, 10, 1)
    assert sample_mapping_data.map_size == (0.9, 0.9, 0.0)
    assert np.allclose(sample_mapping_data.get_spectral_axis(), np.linspace(400, 800, 1000))
    assert sample_mapping_data._get_channel_axis_unit() == "nm"
