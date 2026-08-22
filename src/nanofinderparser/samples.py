"""A catalog of synthetic samples.

Writing a :class:`~nanofinderparser.synthetic.MappingSpec` from scratch means choosing a
spectral range, a handful of peaks and a background that hang together. This module keeps a few
such choices ready to use, each imitating a material that turns up often under a Raman or
photoluminescence microscope, so that a test, an example or a demo can ask for a mapping of
graphene and get spectra that look the part.

Every sample is a function of the same keyword arguments --- the size of the scan, the length of
the spectra, the excitation wavelength and the seed of the noise --- returning a
:class:`~nanofinderparser.synthetic.MappingSpec`. Nothing is hidden in them: the returned spec
can be inspected, or rebuilt with :func:`dataclasses.replace` to change whatever the caller
needs.

Examples
--------
>>> spec = sample_spec("graphene")
>>> [peak.shape for peak in spec.peaks]
['lorentzian', 'lorentzian', 'lorentzian']

>>> mapping = sample_mapping("silicon", x_size=4, y_size=3, n_points=128)
>>> mapping.get_map().shape
(3, 4, 128)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, get_args

from nanofinderparser.models import Mapping
from nanofinderparser.synthetic import (
    BaselineSpec,
    InstrumentSpec,
    MappingSpec,
    MapSpec,
    NoiseSpec,
    PeakSpec,
    SpectralAxisSpec,
    build_mapping,
    map_band,
    map_blob,
    map_disk,
    map_product,
    map_ramp,
)

logger = logging.getLogger(__name__)

# Name of a sample of the catalog.
type SampleName = Literal["graphene", "mos2", "hbn", "silicon", "gaussians", "noise"]

# Signature shared by every builder of the catalog.
type SampleBuilder = Callable[..., MappingSpec]

# Number of points along the fast axis of a sample.
DEFAULT_X_SIZE: Final[int] = 32

# Number of points along the slow axis of a sample.
DEFAULT_Y_SIZE: Final[int] = 24

# Distance between consecutive points of a sample, in nm.
DEFAULT_STEP: Final[float] = 500.0

# Number of points of every spectrum of a sample.
DEFAULT_POINTS: Final[int] = 1024

# Excitation wavelength of a sample, in nm.
DEFAULT_LASER_NM: Final[float] = 532.0

# Seed of the noise, fixed so that a sample always comes out the same.
DEFAULT_SEED: Final[int] = 20260822


@dataclass(frozen=True, slots=True)
class SampleInfo:
    """What a sample of the catalog holds.

    Attributes
    ----------
    name : SampleName
        Key of the sample in :data:`SAMPLES`.
    title : str
        Name of the sample, fit to be shown in a menu.
    description : str
        One line about what the mapping contains, fit to be shown as a tooltip.
    technique : {"raman", "pl"}
        Whether the spectra imitate a Raman or a photoluminescence measurement.
    """

    name: SampleName
    title: str
    description: str
    technique: Literal["raman", "pl"]


def _map(x_size: int, y_size: int, step: float) -> MapSpec:
    """Build the scan of a sample.

    Parameters
    ----------
    x_size, y_size : int
        Number of points along each axis.
    step : float
        Distance between consecutive points, in nm.

    Returns
    -------
    MapSpec
        The scan, with the same step along both axes.
    """
    return MapSpec(x_size=x_size, y_size=y_size, x_step=step, y_step=step)


def _instrument(laser_nm: float, information: str) -> InstrumentSpec:
    """Build the acquisition settings of a sample.

    Parameters
    ----------
    laser_nm : float
        Excitation wavelength, in nm.
    information : str
        Free-text comment written to the header of the file.

    Returns
    -------
    InstrumentSpec
        The settings, plausible rather than measured.
    """
    return InstrumentSpec(
        laser_wavelength_nm=laser_nm,
        laser_power_mw=1.2,
        exposure_time_s=0.5,
        accumulations=2,
        information=information,
    )


def graphene(  # noqa: PLR0913
    *,
    x_size: int = DEFAULT_X_SIZE,
    y_size: int = DEFAULT_Y_SIZE,
    step: float = DEFAULT_STEP,
    n_points: int = DEFAULT_POINTS,
    laser_nm: float = DEFAULT_LASER_NM,
    seed: int | None = DEFAULT_SEED,
) -> MappingSpec:
    """Describe a Raman mapping of a graphene flake.

    The G band softens from left to right, as it does under increasing tensile strain. A round
    bilayer patch sits in the middle of the scan, where the 2D band loses most of its height and
    broadens. A defective edge along the left of the flake lights up the D band. Underneath, a
    photoluminescence background grows towards the top of the map.

    Parameters
    ----------
    x_size, y_size : int, optional
        Number of points of the scan along each axis.
    step : float, optional
        Distance between consecutive points, in nm.
    n_points : int, optional
        Number of points of every spectrum.
    laser_nm : float, optional
        Excitation wavelength, in nm.
    seed : int | None, optional
        Seed of the noise. None draws a different mapping every time.

    Returns
    -------
    MappingSpec
        The description of the mapping, with the D, G and 2D bands as Raman shifts.

    Examples
    --------
    >>> spec = graphene(x_size=4, y_size=3, n_points=256)
    >>> len(spec.peaks)
    3
    """
    return MappingSpec(
        map=_map(x_size, y_size, step),
        # 545-650 nm covers roughly 450-3400 cm-1 for a 532 nm excitation.
        spectral_axis=SpectralAxisSpec(size=n_points, start=545.0, stop=650.0),
        peaks=[
            PeakSpec(
                center=1350.0,
                fwhm=32.0,
                amplitude=map_band(240.0, 6.0, start=-0.06, stop=0.10, softness=0.06),
                shape="lorentzian",
                units="raman_shift",
            ),
            PeakSpec(
                center=map_ramp(1584.0, 1578.0),
                fwhm=16.0,
                amplitude=700.0,
                shape="lorentzian",
                units="raman_shift",
            ),
            PeakSpec(
                center=map_disk(2700.0, 2685.0, radius=0.22),
                fwhm=map_disk(52.0, 30.0, radius=0.22),
                amplitude=map_disk(280.0, 900.0, radius=0.22),
                shape="lorentzian",
                units="raman_shift",
            ),
        ],
        baseline=BaselineSpec(offset=map_ramp(120.0, 200.0, axis="y"), slope=-40.0),
        noise=NoiseSpec(poisson=True, sigma=4.0, seed=seed),
        instrument=_instrument(laser_nm, "Synthetic graphene-like mapping"),
    )


def mos2(  # noqa: PLR0913
    *,
    x_size: int = DEFAULT_X_SIZE,
    y_size: int = DEFAULT_Y_SIZE,
    step: float = DEFAULT_STEP,
    n_points: int = DEFAULT_POINTS,
    laser_nm: float = DEFAULT_LASER_NM,
    seed: int | None = DEFAULT_SEED,
) -> MappingSpec:
    """Describe a photoluminescence mapping of a MoS2 flake.

    A monolayer flake covers most of the scan, save for a strip of bare substrate on the right
    where the emission dies out. The A exciton, near 1.85 eV, dominates, with the weaker and
    broader B exciton near 1.99 eV beside it. Over a round bilayer patch the A exciton is
    quenched several times over and settles slightly to the red, as the gap turns indirect.

    Parameters
    ----------
    x_size, y_size : int, optional
        Number of points of the scan along each axis.
    step : float, optional
        Distance between consecutive points, in nm.
    n_points : int, optional
        Number of points of every spectrum.
    laser_nm : float, optional
        Excitation wavelength, in nm.
    seed : int | None, optional
        Seed of the noise. None draws a different mapping every time.

    Returns
    -------
    MappingSpec
        The description of the mapping, with both excitons given as wavelengths.

    Examples
    --------
    >>> spec = mos2(x_size=4, y_size=3, n_points=256)
    >>> spec.spectral_axis.start, spec.spectral_axis.stop
    (600.0, 720.0)
    """
    # The flake, and the bilayer patch within it.
    flake = map_band(0.03, 1.0, start=0.82, stop=1.06, softness=0.04)
    bilayer = map_disk(0.22, 1.0, cx=0.42, cy=0.55, radius=0.18, softness=0.05)

    return MappingSpec(
        map=_map(x_size, y_size, step),
        spectral_axis=SpectralAxisSpec(size=n_points, start=600.0, stop=720.0),
        peaks=[
            # B exciton, 1.99 eV.
            PeakSpec(
                center=623.0,
                fwhm=22.0,
                amplitude=map_product(260.0, flake),
                shape="gaussian",
            ),
            # A exciton, 1.85 eV, quenched and slightly redshifted over the bilayer.
            PeakSpec(
                center=map_disk(672.5, 670.0, cx=0.42, cy=0.55, radius=0.18),
                fwhm=map_disk(24.0, 18.0, cx=0.42, cy=0.55, radius=0.18),
                amplitude=map_product(1400.0, flake, bilayer),
                shape="pseudo_voigt",
                eta=0.3,
            ),
        ],
        baseline=BaselineSpec(offset=90.0, slope=70.0, curvature=-60.0),
        noise=NoiseSpec(poisson=True, sigma=5.0, seed=seed),
        instrument=_instrument(laser_nm, "Synthetic MoS2-like photoluminescence mapping"),
    )


def hbn(  # noqa: PLR0913
    *,
    x_size: int = DEFAULT_X_SIZE,
    y_size: int = DEFAULT_Y_SIZE,
    step: float = DEFAULT_STEP,
    n_points: int = DEFAULT_POINTS,
    laser_nm: float = DEFAULT_LASER_NM,
    seed: int | None = DEFAULT_SEED,
) -> MappingSpec:
    """Describe a Raman mapping of a hexagonal boron nitride flake.

    The E2g mode near 1366 cm-1 is the only sharp feature. Over a thinner region of the flake it
    weakens, broadens and creeps up by a few wavenumbers. A bright spot of defect emission sits
    off to one side, riding on a curved fluorescence background --- a mapping worth pointing a
    baseline correction at.

    Parameters
    ----------
    x_size, y_size : int, optional
        Number of points of the scan along each axis.
    step : float, optional
        Distance between consecutive points, in nm.
    n_points : int, optional
        Number of points of every spectrum.
    laser_nm : float, optional
        Excitation wavelength, in nm.
    seed : int | None, optional
        Seed of the noise. None draws a different mapping every time.

    Returns
    -------
    MappingSpec
        The description of the mapping, with the E2g mode as a Raman shift and the defect
        emission as a wavelength.

    Examples
    --------
    >>> spec = hbn(x_size=4, y_size=3, n_points=256)
    >>> spec.peaks[0].units
    'raman_shift'
    """
    return MappingSpec(
        map=_map(x_size, y_size, step),
        # 560-600 nm covers roughly 1050-2130 cm-1 for a 532 nm excitation.
        spectral_axis=SpectralAxisSpec(size=n_points, start=560.0, stop=600.0),
        peaks=[
            PeakSpec(
                center=map_disk(1369.5, 1366.0, cx=0.35, cy=0.6, radius=0.22, softness=0.03),
                fwhm=map_disk(14.0, 9.0, cx=0.35, cy=0.6, radius=0.22, softness=0.03),
                amplitude=map_disk(190.0, 540.0, cx=0.35, cy=0.6, radius=0.22, softness=0.03),
                shape="lorentzian",
                units="raman_shift",
            ),
            # Defect emission, broad enough to pass for a background if it is not looked at.
            PeakSpec(
                center=589.0,
                fwhm=26.0,
                amplitude=map_blob(430.0, 25.0, cx=0.75, cy=0.3, radius=0.09),
                shape="gaussian",
            ),
        ],
        baseline=BaselineSpec(offset=150.0, slope=180.0, curvature=-120.0),
        noise=NoiseSpec(poisson=True, sigma=4.0, seed=seed),
        instrument=_instrument(laser_nm, "Synthetic hBN-like mapping"),
    )


def silicon(  # noqa: PLR0913
    *,
    x_size: int = DEFAULT_X_SIZE,
    y_size: int = DEFAULT_Y_SIZE,
    step: float = DEFAULT_STEP,
    n_points: int = DEFAULT_POINTS,
    laser_nm: float = DEFAULT_LASER_NM,
    seed: int | None = DEFAULT_SEED,
) -> MappingSpec:
    """Describe a Raman mapping of a silicon wafer.

    One sharp line at 520.7 cm-1 over a nearly flat background, with little noise: the reference
    a spectrometer is calibrated against. A round region where the line drops by about a
    wavenumber stands for a patch of compressively relaxed material, and gives a peak-fitting
    routine something small to resolve.

    Parameters
    ----------
    x_size, y_size : int, optional
        Number of points of the scan along each axis.
    step : float, optional
        Distance between consecutive points, in nm.
    n_points : int, optional
        Number of points of every spectrum.
    laser_nm : float, optional
        Excitation wavelength, in nm.
    seed : int | None, optional
        Seed of the noise. None draws a different mapping every time.

    Returns
    -------
    MappingSpec
        The description of the mapping, with the line given as a Raman shift.

    Examples
    --------
    >>> spec = silicon(x_size=4, y_size=3, n_points=256)
    >>> spec.peaks[0].fwhm
    3.5
    """
    return MappingSpec(
        map=_map(x_size, y_size, step),
        # 540-556 nm covers roughly 280-810 cm-1 for a 532 nm excitation.
        spectral_axis=SpectralAxisSpec(size=n_points, start=540.0, stop=556.0),
        peaks=[
            PeakSpec(
                center=map_disk(519.6, 520.7, cx=0.6, cy=0.45, radius=0.18, softness=0.04),
                fwhm=3.5,
                amplitude=5000.0,
                shape="lorentzian",
                units="raman_shift",
            )
        ],
        baseline=BaselineSpec(offset=80.0, slope=6.0),
        noise=NoiseSpec(poisson=True, sigma=2.0, seed=seed),
        instrument=_instrument(laser_nm, "Synthetic silicon reference mapping"),
    )


def gaussians(  # noqa: PLR0913
    *,
    x_size: int = DEFAULT_X_SIZE,
    y_size: int = DEFAULT_Y_SIZE,
    step: float = DEFAULT_STEP,
    n_points: int = DEFAULT_POINTS,
    laser_nm: float = DEFAULT_LASER_NM,
    seed: int | None = DEFAULT_SEED,
) -> MappingSpec:
    """Describe three Gaussian bands, each brightest over its own corner of the scan.

    Nothing pretends to be a material here: the bands are far apart, of plain shape, and their
    intensities peak at three separate places, which makes it easy to tell at a glance whether
    a map of a fitted parameter came out where it should.

    Parameters
    ----------
    x_size, y_size : int, optional
        Number of points of the scan along each axis.
    step : float, optional
        Distance between consecutive points, in nm.
    n_points : int, optional
        Number of points of every spectrum.
    laser_nm : float, optional
        Excitation wavelength, in nm.
    seed : int | None, optional
        Seed of the noise. None draws a different mapping every time.

    Returns
    -------
    MappingSpec
        The description of the mapping, with the three bands given as wavelengths.

    Examples
    --------
    >>> spec = gaussians(x_size=4, y_size=3, n_points=256)
    >>> [peak.center for peak in spec.peaks]
    [545.0, 558.0, 570.0]
    """
    return MappingSpec(
        map=_map(x_size, y_size, step),
        spectral_axis=SpectralAxisSpec(size=n_points, start=530.0, stop=580.0),
        peaks=[
            PeakSpec(
                center=545.0,
                fwhm=4.7,
                amplitude=map_blob(800.0, 20.0, cx=0.3, cy=0.3, radius=0.2),
            ),
            PeakSpec(
                center=558.0,
                fwhm=4.7,
                amplitude=map_blob(600.0, 15.0, cx=0.7, cy=0.4, radius=0.2),
            ),
            PeakSpec(
                center=570.0,
                fwhm=4.7,
                amplitude=map_blob(400.0, 10.0, cx=0.5, cy=0.75, radius=0.2),
            ),
        ],
        baseline=BaselineSpec(offset=100.0, slope=10.0),
        noise=NoiseSpec(poisson=True, sigma=3.0, seed=seed),
        instrument=_instrument(laser_nm, "Synthetic three-band mapping"),
    )


def noise(  # noqa: PLR0913
    *,
    x_size: int = DEFAULT_X_SIZE,
    y_size: int = DEFAULT_Y_SIZE,
    step: float = DEFAULT_STEP,
    n_points: int = DEFAULT_POINTS,
    laser_nm: float = DEFAULT_LASER_NM,
    seed: int | None = DEFAULT_SEED,
) -> MappingSpec:
    """Describe a mapping with no peaks at all.

    A curved background that brightens towards the top of the scan, plus shot and read noise.
    Anything a peak finder or a baseline correction reports here is something it invented.

    Parameters
    ----------
    x_size, y_size : int, optional
        Number of points of the scan along each axis.
    step : float, optional
        Distance between consecutive points, in nm.
    n_points : int, optional
        Number of points of every spectrum.
    laser_nm : float, optional
        Excitation wavelength, in nm.
    seed : int | None, optional
        Seed of the noise. None draws a different mapping every time.

    Returns
    -------
    MappingSpec
        The description of the mapping, holding no peaks.

    Examples
    --------
    >>> noise(x_size=4, y_size=3, n_points=256).peaks
    []
    """
    return MappingSpec(
        map=_map(x_size, y_size, step),
        spectral_axis=SpectralAxisSpec(size=n_points, start=500.0, stop=600.0),
        peaks=[],
        baseline=BaselineSpec(offset=map_ramp(80.0, 140.0, axis="y"), slope=60.0, curvature=-90.0),
        noise=NoiseSpec(poisson=True, sigma=6.0, seed=seed),
        instrument=_instrument(laser_nm, "Synthetic background-only mapping"),
    )


# The builder of every sample of the catalog, keyed by name.
BUILDERS: Final[dict[SampleName, SampleBuilder]] = {
    "graphene": graphene,
    "mos2": mos2,
    "hbn": hbn,
    "silicon": silicon,
    "gaussians": gaussians,
    "noise": noise,
}

# What every sample of the catalog holds, keyed by name.
SAMPLES: Final[dict[SampleName, SampleInfo]] = {
    "graphene": SampleInfo(
        name="graphene",
        title="Graphene",
        description="D, G and 2D bands, with strain along x and a bilayer patch in the middle.",
        technique="raman",
    ),
    "mos2": SampleInfo(
        name="mos2",
        title="MoS2 photoluminescence",
        description="A and B excitons of a monolayer flake, quenched over a bilayer patch.",
        technique="pl",
    ),
    "hbn": SampleInfo(
        name="hbn",
        title="Hexagonal boron nitride",
        description="E2g mode over a curved fluorescence background, with a defect-emission spot.",
        technique="raman",
    ),
    "silicon": SampleInfo(
        name="silicon",
        title="Silicon reference",
        description="A single sharp line at 520.7 cm-1, barely any noise.",
        technique="raman",
    ),
    "gaussians": SampleInfo(
        name="gaussians",
        title="Three Gaussian bands",
        description="Three well-separated bands, each brightest over its own part of the scan.",
        technique="raman",
    ),
    "noise": SampleInfo(
        name="noise",
        title="Background only",
        description="No peaks: a curved background and noise, to check what gets invented.",
        technique="raman",
    ),
}


def sample_spec(  # noqa: PLR0913
    name: SampleName,
    *,
    x_size: int = DEFAULT_X_SIZE,
    y_size: int = DEFAULT_Y_SIZE,
    step: float = DEFAULT_STEP,
    n_points: int = DEFAULT_POINTS,
    laser_nm: float = DEFAULT_LASER_NM,
    seed: int | None = DEFAULT_SEED,
) -> MappingSpec:
    """Describe one of the samples of the catalog.

    Parameters
    ----------
    name : SampleName
        Which sample to describe. See :data:`SAMPLES`.
    x_size, y_size : int, optional
        Number of points of the scan along each axis.
    step : float, optional
        Distance between consecutive points, in nm.
    n_points : int, optional
        Number of points of every spectrum.
    laser_nm : float, optional
        Excitation wavelength, in nm.
    seed : int | None, optional
        Seed of the noise. None draws a different mapping every time.

    Returns
    -------
    MappingSpec
        The description of the mapping.

    Raises
    ------
    ValueError
        If the name is not one of the samples of the catalog.

    Examples
    --------
    >>> sample_spec("noise", x_size=2, y_size=2).peaks
    []
    """
    try:
        builder = BUILDERS[name]
    except KeyError:
        known = ", ".join(get_args(SampleName.__value__))
        msg = f"Unknown sample {name!r}; expected one of {known}."
        raise ValueError(msg) from None

    return builder(
        x_size=x_size,
        y_size=y_size,
        step=step,
        n_points=n_points,
        laser_nm=laser_nm,
        seed=seed,
    )


def sample_mapping(  # noqa: PLR0913
    name: SampleName,
    *,
    x_size: int = DEFAULT_X_SIZE,
    y_size: int = DEFAULT_Y_SIZE,
    step: float = DEFAULT_STEP,
    n_points: int = DEFAULT_POINTS,
    laser_nm: float = DEFAULT_LASER_NM,
    seed: int | None = DEFAULT_SEED,
) -> Mapping:
    """Build one of the samples of the catalog, without writing any file.

    Parameters
    ----------
    name : SampleName
        Which sample to build. See :data:`SAMPLES`.
    x_size, y_size : int, optional
        Number of points of the scan along each axis.
    step : float, optional
        Distance between consecutive points, in nm.
    n_points : int, optional
        Number of points of every spectrum.
    laser_nm : float, optional
        Excitation wavelength, in nm.
    seed : int | None, optional
        Seed of the noise. None draws a different mapping every time.

    Returns
    -------
    Mapping
        The mapping, holding the same information a mapping read from an SMD file does.

    Raises
    ------
    ValueError
        If the name is not one of the samples of the catalog.

    Examples
    --------
    >>> mapping = sample_mapping("graphene", x_size=4, y_size=3, n_points=128)
    >>> mapping.map_steps
    (4, 3, 1)
    """
    return build_mapping(
        sample_spec(
            name,
            x_size=x_size,
            y_size=y_size,
            step=step,
            n_points=n_points,
            laser_nm=laser_nm,
            seed=seed,
        )
    )
