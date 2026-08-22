"""Build synthetic NanoFinder mappings and SMD files.

The mappings produced here are described declaratively: the size of the scan, the spectral
axis, a list of peaks, a baseline, and the noise. Every peak property can be a single number, a
``(y_size, x_size)`` array, or a function of the map coordinates, so the spectra can change
across the scan in whatever way the test or the example needs.

The result is either a :class:`~nanofinderparser.models.Mapping` --- the same object
:func:`~nanofinderparser.load.load_smd` returns --- or an actual ``.smd`` file written through
:func:`~nanofinderparser.write.write_smd`.

Writing a spec from scratch is not always needed: :mod:`nanofinderparser.samples` keeps a few
ready-made ones, each imitating a material that turns up often under the microscope. The
``map_*`` functions below draw the simple shapes those samples are built from --- a gradient, a
bump, a patch, a stripe --- and are addressed in fractions of the scanned area, so a spec keeps
its look whatever the size of the map.

Examples
--------
A 20 x 15 mapping with one Gaussian peak whose position drifts along x and whose intensity is
brightest in the middle of the scan:

>>> import numpy as np
>>> from nanofinderparser.synthetic import (
...     BaselineSpec,
...     MapSpec,
...     MappingSpec,
...     NoiseSpec,
...     PeakSpec,
...     SpectralAxisSpec,
...     create_smd,
... )
>>> spec = MappingSpec(
...     map=MapSpec(x_size=20, y_size=15, x_step=250.0, y_step=250.0),
...     spectral_axis=SpectralAxisSpec(size=512, start=540.0, stop=600.0),
...     peaks=[
...         PeakSpec(
...             center=lambda x, y: 1580.0 + 20.0 * x / x.max(),
...             fwhm=15.0,
...             amplitude=lambda x, y: 800.0 * np.exp(-(((x - 2500) / 1500) ** 2)),
...             shape="lorentzian",
...             units="raman_shift",
...         )
...     ],
...     baseline=BaselineSpec(offset=100.0, slope=20.0),
...     noise=NoiseSpec(poisson=True, sigma=5.0, seed=0),
... )
>>> create_smd("synthetic.smd", spec)  # doctest: +SKIP
PosixPath('synthetic.smd')

"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from nanofinderparser.models import (
    SMD_DATE_FORMAT,
    SMD_TIME_FORMAT,
    Channel,
    ChannelInfo,
    Mapping,
)
from nanofinderparser.units import Units, convert_spectral_units
from nanofinderparser.write import DEFAULT_VENDOR, NIL_GUID, write_smd

logger = logging.getLogger(__name__)

# Timestamp written when the spec does not give one. It is fixed rather than taken from the
# clock so that the same spec always produces the same file.
DEFAULT_MEASURED_AT: Final[datetime] = datetime(2024, 1, 1, 12, 0, 0)  # noqa: DTZ001

# Shape of a peak.
type PeakShape = Literal["gaussian", "lorentzian", "pseudo_voigt"]

# A quantity that may change from one point of the map to another. It can be given as a single
# number, as an array broadcastable to ``(y_size, x_size)``, or as a function of the physical map
# coordinates, which arrive as two ``(y_size, x_size)`` arrays in the units of the stage axes.
type MapParameter = (
    float
    | Sequence[float]
    | Sequence[Sequence[float]]
    | NDArray[np.float64]
    | Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64] | float]
)

# Peak width below which a lineshape would collapse to a spike.
_MIN_FWHM: Final[float] = 1e-12

# Radius below which a shape drawn over the map would collapse to a point.
_MIN_RADIUS: Final[float] = 1e-12


def _evaluate(
    value: MapParameter,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    name: str,
) -> NDArray[np.float64]:
    """Turn a map parameter into a ``(y_size, x_size)`` array.

    Parameters
    ----------
    value : MapParameter
        A number, an array broadcastable to the shape of the map, or a callable of the map
        coordinates.
    x, y : NDArray[np.float64]
        Physical coordinates of every point of the map, both of shape ``(y_size, x_size)``.
    name : str
        Name of the parameter, used in the error message.

    Returns
    -------
    NDArray[np.float64]
        The parameter evaluated at every point of the map.

    Raises
    ------
    ValueError
        If the value cannot be broadcast to the shape of the map.
    """
    raw = value(x, y) if callable(value) else value
    array = np.asarray(raw, dtype=np.float64)

    try:
        return np.broadcast_to(array, x.shape)
    except ValueError:
        msg = (
            f"The {name!r} parameter has shape {array.shape}, which cannot be broadcast to the "
            f"{x.shape} shape of the map."
        )
        raise ValueError(msg) from None


# ---------------------------------------------------------------------------
# Shapes
#
# Factories returning map parameters that draw a simple shape over the scanned area. They are
# addressed in fractions of the extent of the map -- 0 at the first point of an axis, 1 at the
# last one -- so that a spec keeps its look when the size or the step of the scan changes.
# ---------------------------------------------------------------------------


def _extent(values: NDArray[np.float64]) -> tuple[float, float]:
    """Origin and length of a coordinate grid.

    Parameters
    ----------
    values : NDArray[np.float64]
        Coordinates of every point of the map along one axis.

    Returns
    -------
    origin : float
        Smallest coordinate.
    length : float
        Distance between the first and the last point, zero for a single-point axis.
    """
    return float(values.min()), float(np.ptp(values))


def _fraction(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Express a coordinate grid as a fraction of its own extent.

    Parameters
    ----------
    values : NDArray[np.float64]
        Coordinates of every point of the map along one axis.

    Returns
    -------
    NDArray[np.float64]
        Zero at the first point of the axis, one at the last one. All zeros when the axis holds
        a single point.
    """
    origin, length = _extent(values)
    if length == 0.0:
        return np.zeros_like(values)
    return (values - origin) / length


def _smoothstep(edge: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ease a transition between 0 and 1.

    Parameters
    ----------
    edge : NDArray[np.float64]
        How far past the edge every point lies, in units of the width of the transition.

    Returns
    -------
    NDArray[np.float64]
        0 before the edge, 1 after it, and a smooth ramp with zero slope at both ends between
        them.
    """
    clipped = np.clip(edge, 0.0, 1.0)
    eased: NDArray[np.float64] = clipped**2 * (3.0 - 2.0 * clipped)
    return eased


def map_ramp(
    low: float,
    high: float,
    *,
    axis: Literal["x", "y"] = "x",
) -> MapParameter:
    """Draw a value changing linearly across the map.

    Parameters
    ----------
    low, high : float
        Value at the first and at the last point of `axis`.
    axis : {"x", "y"}, optional
        Direction along which the value changes, by default ``"x"``.

    Returns
    -------
    MapParameter
        A map parameter usable anywhere in a :class:`MappingSpec`.

    Examples
    --------
    >>> strain = map_ramp(1582.0, 1577.0)
    >>> spec = MappingSpec(
    ...     map=MapSpec(x_size=3, y_size=2),
    ...     peaks=[PeakSpec(center=strain, fwhm=5.0, amplitude=100.0)],
    ... )
    >>> build_spectra(spec).shape
    (2, 3, 512)
    """

    def shape(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        fraction = _fraction(x if axis == "x" else y)
        return low + (high - low) * fraction

    return shape


def map_blob(
    value: float,
    background: float = 0.0,
    *,
    cx: float = 0.5,
    cy: float = 0.5,
    radius: float = 0.2,
) -> MapParameter:
    """Draw a round Gaussian bump over the map.

    Parameters
    ----------
    value : float
        Value at the center of the bump.
    background : float, optional
        Value far away from it, by default 0.0.
    cx, cy : float, optional
        Center of the bump, as a fraction of the extent of each axis, by default the middle of
        the map.
    radius : float, optional
        Standard deviation of the bump, as a fraction of the longest extent of the map, by
        default 0.2.

    Returns
    -------
    MapParameter
        A map parameter usable anywhere in a :class:`MappingSpec`.
    """

    def shape(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        distance = _distance(x, y, cx, cy)
        weight = np.exp(-0.5 * (distance / max(radius, _MIN_RADIUS)) ** 2)
        return background + (value - background) * weight

    return shape


def map_disk(  # noqa: PLR0913
    inside: float,
    outside: float,
    *,
    cx: float = 0.5,
    cy: float = 0.5,
    radius: float = 0.25,
    softness: float = 0.0,
) -> MapParameter:
    """Draw a round patch over the map.

    Parameters
    ----------
    inside, outside : float
        Value within and beyond the patch.
    cx, cy : float, optional
        Center of the patch, as a fraction of the extent of each axis, by default the middle of
        the map.
    radius : float, optional
        Radius of the patch, as a fraction of the longest extent of the map, by default 0.25.
    softness : float, optional
        Width of the transition at the edge of the patch, in the same fractional units, by
        default 0.0, which gives a sharp edge.

    Returns
    -------
    MapParameter
        A map parameter usable anywhere in a :class:`MappingSpec`.
    """

    def shape(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        distance = _distance(x, y, cx, cy)
        if softness <= 0.0:
            return np.where(distance <= radius, inside, outside)
        weight = _smoothstep((radius - distance) / softness + 0.5)
        return outside + (inside - outside) * weight

    return shape


def map_band(  # noqa: PLR0913
    inside: float,
    outside: float,
    *,
    start: float = 0.0,
    stop: float = 0.15,
    axis: Literal["x", "y"] = "x",
    softness: float = 0.0,
) -> MapParameter:
    """Draw a stripe running across the map.

    Parameters
    ----------
    inside, outside : float
        Value within and beyond the stripe.
    start, stop : float
        Limits of the stripe along `axis`, as a fraction of its extent.
    axis : {"x", "y"}, optional
        Direction along which the stripe is delimited, by default ``"x"``, which gives a stripe
        parallel to the y axis.
    softness : float, optional
        Width of the transition at the edges of the stripe, in the same fractional units, by
        default 0.0, which gives sharp edges. The transition is centered on each limit, so half
        of it falls inside the stripe. Put a limit beyond the map, as in ``start=-0.05``, for a
        stripe that reaches an edge of the scan at full value.

    Returns
    -------
    MapParameter
        A map parameter usable anywhere in a :class:`MappingSpec`.
    """

    def shape(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        fraction = _fraction(x if axis == "x" else y)
        if softness <= 0.0:
            return np.where((fraction >= start) & (fraction <= stop), inside, outside)
        weight = _smoothstep((fraction - start) / softness + 0.5) * _smoothstep(
            (stop - fraction) / softness + 0.5
        )
        return outside + (inside - outside) * weight

    return shape


def map_product(*parts: MapParameter) -> MapParameter:
    """Multiply several map parameters together.

    Handy to modulate one shape by another, for instance to dim a peak over a patch of a flake
    that does not cover the whole scanned area.

    Parameters
    ----------
    *parts : MapParameter
        The parameters to multiply. Each of them may be a number, an array, or a callable.

    Returns
    -------
    MapParameter
        A map parameter worth the product of all the parts at every point of the map.

    Examples
    --------
    >>> flake = map_disk(1.0, 0.0, radius=0.4)
    >>> bilayer = map_disk(0.3, 1.0, radius=0.15)
    >>> amplitude = map_product(900.0, flake, bilayer)
    """

    def shape(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.ones_like(x)
        for index, part in enumerate(parts):
            result = result * _evaluate(part, x, y, f"factor {index}")
        return result

    return shape


def _distance(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    cx: float,
    cy: float,
) -> NDArray[np.float64]:
    """Distance from every point of the map to a point given in fractional coordinates.

    Parameters
    ----------
    x, y : NDArray[np.float64]
        Physical coordinates of every point of the map.
    cx, cy : float
        The point of interest, as a fraction of the extent of each axis.

    Returns
    -------
    NDArray[np.float64]
        The distance, as a fraction of the longest extent of the map, so that a shape built on
        it stays round whatever the aspect ratio of the scan.
    """
    x_origin, x_length = _extent(x)
    y_origin, y_length = _extent(y)
    scale = max(x_length, y_length) or 1.0

    offset_x = (x - (x_origin + cx * x_length)) / scale
    offset_y = (y - (y_origin + cy * y_length)) / scale
    distance: NDArray[np.float64] = np.hypot(offset_x, offset_y)
    return distance


@dataclass(frozen=True, slots=True)
class MapSpec:
    """Size and stage coordinates of a synthetic scan.

    Attributes
    ----------
    x_size : int
        Number of points along the x axis, which is the fast one.
    y_size : int
        Number of points along the y axis, which is the slow one.
    x_step, y_step : float
        Distance between consecutive points, in `units`.
    x_start, y_start : float
        Absolute position of the first point of the scan, in `units`.
    units : str
        Units of the stage axes. NanoFinder writes ``"nm"``.

    Notes
    -----
    Real SMD files store stage positions as integer DAC counts scaled by a per-axis factor, so
    the step size they can represent is quantized. Synthetic files sidestep that by using one
    count per step and putting the step size in the scale factor, which keeps `x_step` and
    `x_start` exact when the file is read back.
    """

    x_size: int
    y_size: int
    x_step: float = 500.0
    y_step: float = 500.0
    x_start: float = 0.0
    y_start: float = 0.0
    units: str = "nm"

    def coordinates(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Physical coordinates of every point of the map.

        Returns
        -------
        x, y : NDArray[np.float64]
            Two arrays of shape ``(y_size, x_size)``, in the units of the stage axes.
        """
        x = self.x_start + np.arange(self.x_size, dtype=np.float64) * self.x_step
        y = self.y_start + np.arange(self.y_size, dtype=np.float64) * self.y_step
        x_grid, y_grid = np.meshgrid(x, y)
        return x_grid, y_grid


@dataclass(frozen=True, slots=True)
class SpectralAxisSpec:
    """Spectral axis shared by every spectrum of a synthetic mapping.

    Attributes
    ----------
    size : int
        Number of points of each spectrum.
    start, stop : float
        First and last value of the axis, in `units`.
    units : {"nm", "cm-1", "eV"}
        Units of the axis as stored in the file. NanoFinder writes ``"nm"``.
    values : NDArray[np.float64] | None
        Explicit axis values, which take precedence over `size`, `start` and `stop`.
    name : str
        Name of the axis, as stored in the file.
    """

    size: int = 512
    start: float = 500.0
    stop: float = 600.0
    units: Literal["nm", "cm-1", "eV"] = "nm"
    values: NDArray[np.float64] | None = None
    name: str = "Wavelength"

    def build(self) -> NDArray[np.float64]:
        """Return the values of the spectral axis.

        Returns
        -------
        NDArray[np.float64]
            The axis, of shape ``(size,)``, in `units`.

        Raises
        ------
        ValueError
            If the axis holds fewer than one point, or if `values` is not one-dimensional.
        """
        if self.values is not None:
            axis = np.asarray(self.values, dtype=np.float64)
            if axis.ndim != 1:
                msg = f"The spectral axis must be one-dimensional, but its shape is {axis.shape}."
                raise ValueError(msg)
        else:
            axis = np.linspace(self.start, self.stop, self.size, dtype=np.float64)

        if axis.size < 1:
            msg = "The spectral axis must hold at least one point."
            raise ValueError(msg)

        return axis


@dataclass(frozen=True, slots=True)
class PeakSpec:
    """A peak, possibly changing from one point of the map to another.

    Attributes
    ----------
    center : MapParameter
        Position of the maximum, in `units`.
    fwhm : MapParameter
        Full width at half maximum, in `units`.
    amplitude : MapParameter
        Height of the peak above the baseline, in counts.
    shape : {"gaussian", "lorentzian", "pseudo_voigt"}
        Lineshape of the peak.
    eta : MapParameter
        Weight of the Lorentzian part of a pseudo-Voigt, between 0 and 1. Ignored by the other
        shapes.
    units : {"nm", "cm-1", "eV", "raman_shift"} | None
        Units in which `center` and `fwhm` are given. When None they follow the units of the
        spectral axis. Any other choice converts the axis before the peak is evaluated, so a
        Raman peak keeps its width in cm-1 across the whole axis.
    """

    center: MapParameter
    fwhm: MapParameter
    amplitude: MapParameter
    shape: PeakShape = "gaussian"
    eta: MapParameter = 0.5
    units: Units | Literal["nm", "cm-1", "eV", "raman_shift"] | None = None


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    """Smooth background added under the peaks.

    The baseline is evaluated as ``offset + slope * t + curvature * t**2``, where ``t`` runs
    from 0 at the first point of the spectral axis to 1 at the last one. Working on that
    normalized axis keeps the coefficients independent of the units and of the range of the
    axis.

    Attributes
    ----------
    offset : MapParameter
        Value of the baseline at the first point of the spectrum, in counts.
    slope : MapParameter
        Linear term, in counts across the whole spectral range.
    curvature : MapParameter
        Quadratic term, in counts across the whole spectral range.
    """

    offset: MapParameter = 0.0
    slope: MapParameter = 0.0
    curvature: MapParameter = 0.0


@dataclass(frozen=True, slots=True)
class NoiseSpec:
    """Noise added to the spectra.

    Attributes
    ----------
    poisson : bool
        Whether to draw every value from a Poisson distribution, which is the shot noise of a
        photon-counting detector. Values are clipped at zero beforehand, and the result is a
        whole number of counts.
    sigma : MapParameter
        Standard deviation of an additional Gaussian noise, in counts, standing for the read
        noise of the detector. Zero disables it.
    seed : int | None
        Seed of the random generator. Passing one makes the mapping reproducible; None draws a
        different mapping every time.
    """

    poisson: bool = False
    sigma: MapParameter = 0.0
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Acquisition settings written to the header of the file.

    None of these affect the generated spectra: they fill in the metadata a real SMD file
    carries, so that code reading the synthetic file finds what it expects.

    Attributes
    ----------
    laser_wavelength_nm : float
        Excitation wavelength, in nm. Also used to convert peaks given as a Raman shift.
    laser_power_mw : float
        Power of the laser.
    grating_groove : str
        Groove density of the grating.
    central_wavelength_nm : float | None
        Wavelength at the center of the detector. None takes the middle of the spectral axis.
    pinhole_size_um : float
        Size of the pinhole, in micrometers.
    exposure_time_s : float
        Exposure time of a single acquisition, in seconds.
    cycle_time_s : float | None
        Time between consecutive acquisitions, in seconds. None reuses `exposure_time_s`.
    accumulations : int
        Number of accumulated acquisitions per point.
    temperature_c : float
        Temperature of the detector, in degrees Celsius.
    system_name, positioning_sys_name, detection_sys_name : str
        Names of the instrument and of its subsystems.
    information : str
        Free-text comment.
    measured_at : datetime
        Date and time of the measurement. Fixed by default, so that the same spec always
        produces the same file.
    head_model : str
        Model of the detector head.
    ccd_width, ccd_height : int
        Size of the detector, in pixels.
    central_pixel : int
        Pixel used to calibrate the spectral axis.
    pixel_size_um : float
        Size of a detector pixel, in micrometers.
    horizontal_binning : int
        Binning applied along the spectral direction.
    center_row, track_height : int
        Position and height of the readout track on the detector.
    readout_mode : str
        Readout mode of the detector.
    data_channel_name, data_channel_unit : str
        Name and unit of the measured quantity.
    """

    laser_wavelength_nm: float = 532.0
    laser_power_mw: float = 1.0
    grating_groove: str = "600"
    central_wavelength_nm: float | None = None
    pinhole_size_um: float = 50.0
    exposure_time_s: float = 1.0
    cycle_time_s: float | None = None
    accumulations: int = 1
    temperature_c: float = -60.0
    system_name: str = "TII Nanofinder"
    positioning_sys_name: str = "XYZ NI-DACmx"
    detection_sys_name: str = "Andor CCD"
    information: str = "Synthetic data"
    measured_at: datetime = DEFAULT_MEASURED_AT
    head_model: str = "DV420"
    ccd_width: int = 1024
    ccd_height: int = 255
    central_pixel: int = 510
    pixel_size_um: float = 26.0
    horizontal_binning: int = 1
    center_row: int = 53
    track_height: int = 20
    readout_mode: str = "Single Track"
    data_channel_name: str = "Photons"
    data_channel_unit: str = "Counts"


@dataclass(frozen=True, slots=True)
class MappingSpec:
    """Complete description of a synthetic mapping.

    Attributes
    ----------
    map : MapSpec
        Size and stage coordinates of the scan.
    spectral_axis : SpectralAxisSpec
        Spectral axis shared by every spectrum.
    peaks : Sequence[PeakSpec]
        Peaks to add on top of the baseline. May be empty.
    baseline : BaselineSpec
        Smooth background under the peaks.
    noise : NoiseSpec
        Noise added to the spectra.
    instrument : InstrumentSpec
        Acquisition settings written to the header of the file.
    """

    map: MapSpec
    spectral_axis: SpectralAxisSpec = field(default_factory=SpectralAxisSpec)
    peaks: Sequence[PeakSpec] = ()
    baseline: BaselineSpec = field(default_factory=BaselineSpec)
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    instrument: InstrumentSpec = field(default_factory=InstrumentSpec)


def _peak_axis(spec: MappingSpec, axis: NDArray[np.float64], peak: PeakSpec) -> NDArray[np.float64]:
    """Express the spectral axis in the units in which a peak is defined.

    Parameters
    ----------
    spec : MappingSpec
        The mapping being built, which carries the excitation wavelength.
    axis : NDArray[np.float64]
        The spectral axis, in the units of the file.
    peak : PeakSpec
        The peak about to be evaluated.

    Returns
    -------
    NDArray[np.float64]
        The axis in the units of the peak. Evaluating the lineshape there, rather than
        converting its center and width, keeps the peak symmetric in its own units.
    """
    if peak.units is None:
        return axis

    return convert_spectral_units(
        axis,
        spec.spectral_axis.units,
        peak.units,
        laser_wavelength_nm=spec.instrument.laser_wavelength_nm,
    )


def _gaussian(squared: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate a Gaussian lineshape of unit height.

    Parameters
    ----------
    squared : NDArray[np.float64]
        Squared distance to the center of the peak, in half-widths.

    Returns
    -------
    NDArray[np.float64]
        The lineshape, worth 1 at the center of the peak and 0.5 half a width away from it.
    """
    profile: NDArray[np.float64] = np.exp(-np.log(2.0) * squared)
    return profile


def _lorentzian(squared: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate a Lorentzian lineshape of unit height.

    Parameters
    ----------
    squared : NDArray[np.float64]
        Squared distance to the center of the peak, in half-widths.

    Returns
    -------
    NDArray[np.float64]
        The lineshape, worth 1 at the center of the peak and 0.5 half a width away from it.
    """
    profile: NDArray[np.float64] = 1.0 / (1.0 + squared)
    return profile


def _profile(
    shape: str,
    offset: NDArray[np.float64],
    fwhm: NDArray[np.float64],
    eta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate a normalized lineshape, whose maximum is 1.

    Parameters
    ----------
    shape : {"gaussian", "lorentzian", "pseudo_voigt"}
        The lineshape to evaluate. Typed as a plain string, rather than as
        :data:`PeakShape`, so that an unknown name coming from unchecked code is rejected here
        rather than sailing through.
    offset : NDArray[np.float64]
        Distance to the center of the peak, of shape ``(y_size, x_size, n_points)``.
    fwhm : NDArray[np.float64]
        Full width at half maximum, broadcastable to the shape of `offset`.
    eta : NDArray[np.float64]
        Weight of the Lorentzian part of a pseudo-Voigt.

    Returns
    -------
    NDArray[np.float64]
        The lineshape, of the shape of `offset`.

    Raises
    ------
    ValueError
        If the shape is not one of the supported ones.
    """
    reduced = 2.0 * offset / np.maximum(fwhm, _MIN_FWHM)
    squared: NDArray[np.float64] = reduced**2

    if shape == "gaussian":
        return _gaussian(squared)
    if shape == "lorentzian":
        return _lorentzian(squared)
    if shape == "pseudo_voigt":
        blended: NDArray[np.float64] = eta * _lorentzian(squared) + (1.0 - eta) * _gaussian(squared)
        return blended

    msg = f"Unknown peak shape {shape!r}; expected 'gaussian', 'lorentzian' or 'pseudo_voigt'."
    raise ValueError(msg)


def build_spectra(spec: MappingSpec) -> NDArray[np.float32]:
    """Build the spectra of a synthetic mapping.

    Parameters
    ----------
    spec : MappingSpec
        The description of the mapping.

    Returns
    -------
    NDArray[np.float32]
        The spectra, of shape ``(y_size, x_size, n_points)``, laid out like
        :meth:`~nanofinderparser.models.Mapping.get_map`. The values are ``float32``, which is
        what SMD files store.

    Raises
    ------
    ValueError
        If a parameter cannot be broadcast to the shape of the map, or if a peak has an unknown
        shape.

    Examples
    --------
    >>> spec = MappingSpec(map=MapSpec(x_size=3, y_size=2), baseline=BaselineSpec(offset=10.0))
    >>> build_spectra(spec).shape
    (2, 3, 512)
    """
    axis = spec.spectral_axis.build()
    x, y = spec.map.coordinates()

    normalized = (
        np.zeros_like(axis)
        if axis.size == 1
        else (axis - axis[0]) / (axis[-1] - axis[0] if axis[-1] != axis[0] else 1.0)
    )

    offset = _evaluate(spec.baseline.offset, x, y, "baseline offset")[..., None]
    slope = _evaluate(spec.baseline.slope, x, y, "baseline slope")[..., None]
    curvature = _evaluate(spec.baseline.curvature, x, y, "baseline curvature")[..., None]

    intensities = offset + slope * normalized + curvature * normalized**2
    intensities = np.broadcast_to(intensities, (*x.shape, axis.size)).astype(np.float64)

    for index, peak in enumerate(spec.peaks):
        peak_axis = _peak_axis(spec, axis, peak)
        center = _evaluate(peak.center, x, y, f"center of peak {index}")[..., None]
        fwhm = _evaluate(peak.fwhm, x, y, f"fwhm of peak {index}")[..., None]
        amplitude = _evaluate(peak.amplitude, x, y, f"amplitude of peak {index}")[..., None]
        eta = _evaluate(peak.eta, x, y, f"eta of peak {index}")[..., None]

        intensities += amplitude * _profile(peak.shape, peak_axis - center, fwhm, eta)

    return _add_noise(intensities, spec, x, y).astype(np.float32)


def _add_noise(
    intensities: NDArray[np.float64],
    spec: MappingSpec,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Add shot and read noise to the spectra.

    Parameters
    ----------
    intensities : NDArray[np.float64]
        The noiseless spectra, of shape ``(y_size, x_size, n_points)``.
    spec : MappingSpec
        The description of the mapping, which carries the noise settings.
    x, y : NDArray[np.float64]
        Physical coordinates of every point of the map.

    Returns
    -------
    NDArray[np.float64]
        The spectra with noise. The array may be the one passed in, modified in place.
    """
    noise = spec.noise
    sigma = _evaluate(noise.sigma, x, y, "noise sigma")[..., None]

    if not noise.poisson and not np.any(sigma):
        return intensities

    rng = np.random.default_rng(noise.seed)

    if noise.poisson:
        intensities = rng.poisson(np.clip(intensities, 0.0, None)).astype(np.float64)

    if np.any(sigma):
        intensities = intensities + rng.normal(0.0, 1.0, intensities.shape) * sigma

    return intensities


def _build_channel(spec: MappingSpec, axis: NDArray[np.float64]) -> Channel:
    """Build the detector channel of a synthetic mapping.

    Parameters
    ----------
    spec : MappingSpec
        The description of the mapping.
    axis : NDArray[np.float64]
        The spectral axis, in the units of the file.

    Returns
    -------
    Channel
        The single detector channel of the mapping.
    """
    instrument = spec.instrument
    return Channel(
        DeviceGuid=NIL_GUID,
        DeviceName=instrument.detection_sys_name,
        DataChannelName=instrument.data_channel_name,
        DataChannelUnit=instrument.data_channel_unit,
        ChannelSize=int(axis.size),
        ChannelAxisName=spec.spectral_axis.name,
        ChannelAxisUnit=spec.spectral_axis.units,
        ChannelAxisLaserWl=instrument.laser_wavelength_nm,
        ChannelAxisArray=axis,
        SeriesSize=1,
        ChannelInfo=ChannelInfo(
            Temperature=instrument.temperature_c,
            ExposureTime=instrument.exposure_time_s,
            CycleTime=(
                instrument.cycle_time_s
                if instrument.cycle_time_s is not None
                else instrument.exposure_time_s
            ),
            AcquisitionMode="accumulate",
            AccumulationNumber=instrument.accumulations,
            HeadModel=instrument.head_model,
            CcdWidth=instrument.ccd_width,
            CcdHeight=instrument.ccd_height,
            CentralPixel=instrument.central_pixel,
            PixelSizeUm=instrument.pixel_size_um,
            HorizontalBinning=instrument.horizontal_binning,
            CenterRow=instrument.center_row,
            TrackHeight=instrument.track_height,
            ReadoutMode=instrument.readout_mode,
        ),
    )


def _axis_dict(  # noqa: PLR0913
    name: str,
    unit_name: str,
    *,
    start: float,
    step: float,
    in_use: bool,
    is_slow: bool = False,
) -> dict[str, Any]:
    """Build the parameters of one stage axis.

    Parameters
    ----------
    name : str
        Name of the axis.
    unit_name : str
        Units of the axis.
    start : float
        Absolute position of the first point of the scan.
    step : float
        Distance between consecutive points.
    in_use : bool
        Whether the axis takes part in the scan.
    is_slow : bool, optional
        Whether the axis is the slow one of the raster, by default False.

    Returns
    -------
    dict[str, Any]
        The axis parameters, keyed by the names used in the file.

    Notes
    -----
    One DAC count per step, with the step size held by the scale factor, makes the physical
    positions exact instead of quantized.
    """
    return {
        "AxisIsInUse": -1 if in_use else 0,
        "AxisIsInversed": 0,
        "AxisIsSlow": -1 if is_slow else 0,
        "AxisName": name,
        "AxisUnitName": unit_name,
        "AxisCountStart": 0,
        "AxisCountStep": 1,
        "AxisBiasFloat": start,
        "AxisScaleFloat": step,
    }


def build_mapping(spec: MappingSpec, *, source: Path | None = None) -> Mapping:
    """Build a synthetic mapping, without writing any file.

    Parameters
    ----------
    spec : MappingSpec
        The description of the mapping.
    source : Path | None, optional
        Path to record as the origin of the mapping, by default None.

    Returns
    -------
    Mapping
        The mapping, holding the same information a mapping read from an SMD file does.

    Raises
    ------
    ValueError
        If a parameter cannot be broadcast to the shape of the map, or if a peak has an unknown
        shape.

    Examples
    --------
    >>> spec = MappingSpec(map=MapSpec(x_size=4, y_size=3))
    >>> mapping = build_mapping(spec)
    >>> mapping.map_steps
    (4, 3, 1)
    >>> mapping.get_map().shape
    (3, 4, 512)
    """
    axis = spec.spectral_axis.build()
    spectra = build_spectra(spec)
    instrument = spec.instrument
    map_spec = spec.map

    central_wavelength = instrument.central_wavelength_nm
    if central_wavelength is None:
        in_nm = convert_spectral_units(
            axis,
            spec.spectral_axis.units,
            Units.nm,
            laser_wavelength_nm=instrument.laser_wavelength_nm,
        )
        central_wavelength = float(np.mean(in_nm))

    init_dict: dict[str, Any] = {
        "Vendor": DEFAULT_VENDOR,
        "Version": "1",
        "ScannedFrameParameters": {
            "Vendor": DEFAULT_VENDOR,
            "Version": "2",
            "ScanRepeatNumber": 1,
            "FrameHeader": {
                "Vendor": DEFAULT_VENDOR,
                "Version": "1",
                "Date": instrument.measured_at.strftime(SMD_DATE_FORMAT),
                "Time": instrument.measured_at.strftime(SMD_TIME_FORMAT),
                "Information": instrument.information,
                "SystemName": instrument.system_name,
                "PositioningSysName": instrument.positioning_sys_name,
                "DetectionSysName": instrument.detection_sys_name,
                "ScannedDataName": "Mapping",
            },
            "FrameOptions": {
                "Vendor": DEFAULT_VENDOR,
                "Version": "1",
                "OmuLaserWLnm": instrument.laser_wavelength_nm,
                "OmuCurPower": instrument.laser_power_mw,
                "OmuGratingGroove": instrument.grating_groove,
                "OmuCentralWaveLengthNM": central_wavelength,
                "OmuPinHoleSize": instrument.pinhole_size_um,
            },
            "Stage3DParameters": {
                "Vendor": DEFAULT_VENDOR,
                "Version": "1",
                "AxisSizeX": map_spec.x_size,
                "AxisSizeY": map_spec.y_size,
                "AxisSizeZ": 1,
                "StageAxesDimentions": {
                    "AxisX": _axis_dict(
                        "X",
                        map_spec.units,
                        start=map_spec.x_start,
                        step=map_spec.x_step,
                        in_use=True,
                    ),
                    "AxisY": _axis_dict(
                        "Y",
                        map_spec.units,
                        start=map_spec.y_start,
                        step=map_spec.y_step,
                        in_use=True,
                    ),
                    "AxisZ": _axis_dict("Z", map_spec.units, start=0.0, step=1.0, in_use=False),
                },
            },
            "DataCalibration": {
                "Vendor": DEFAULT_VENDOR,
                "Version": "1",
                "Channels": [_build_channel(spec, axis)],
            },
            "OriginalFileName": str(source) if source is not None else None,
            "DataBlockSizeBytes": int(spectra.nbytes),
        },
        "Data": spectra.ravel(),
    }

    return Mapping(init_dict, source=source)


def create_smd(file: Path | str, spec: MappingSpec) -> Path:
    """Build a synthetic mapping and write it as an SMD file.

    Parameters
    ----------
    file : Path | str
        Path of the file to write. Parent directories are created when missing.
    spec : MappingSpec
        The description of the mapping.

    Returns
    -------
    Path
        The path of the file just written.

    Raises
    ------
    ValueError
        If a parameter cannot be broadcast to the shape of the map, or if a peak has an unknown
        shape.
    OSError
        If the file cannot be written.

    Examples
    --------
    >>> from nanofinderparser import load_smd
    >>> spec = MappingSpec(map=MapSpec(x_size=4, y_size=3))
    >>> path = create_smd("synthetic.smd", spec)  # doctest: +SKIP
    >>> load_smd(path).map_steps  # doctest: +SKIP
    (4, 3, 1)
    """
    file = Path(file)
    return write_smd(build_mapping(spec, source=file), file)
