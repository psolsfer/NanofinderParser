"""Create a synthetic NanoFinder SMD file and read it back.

Synthetic files are useful when a real mapping is too big to commit, too slow to load, or simply
does not exist yet: tests, examples and benchmarks can all be built from a spec instead. This
script writes one, reads it back with the normal parser, and prints what came out of it.

The mapping imitates a Raman map of graphene: a G peak whose position drifts with strain across
the sample, and a 2D peak that dims over a round bilayer patch in the middle. Everything is
generated from the spec below, so changing a number changes the file.

Run it from the project root:

    python scripts/create_synthetic_smd.py
    python scripts/create_synthetic_smd.py my_mapping.smd
"""

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nanofinderparser import (
    BaselineSpec,
    InstrumentSpec,
    MappingSpec,
    MapSpec,
    NoiseSpec,
    PeakSpec,
    SpectralAxisSpec,
    create_smd,
    load_smd,
)
from nanofinderparser.models import Mapping

# ruff: noqa: T201

DEFAULT_FILE = Path("synthetic_mapping.smd")

X_SIZE, Y_SIZE = 24, 18
STEP_NM = 500.0
X_EXTENT = (X_SIZE - 1) * STEP_NM
Y_EXTENT = (Y_SIZE - 1) * STEP_NM

LASER_NM = 532.0
BILAYER_RADIUS_NM = 2500.0

# Raman shifts, in cm-1, of the two bands of the fictional sample.
G_BAND = (1500.0, 1700.0)
TWO_D_BAND = (2550.0, 2800.0)

SHADES = " .:-=+*#%@"


def g_center(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Position of the G peak, softening from left to right as if the sample were strained.

    Parameters
    ----------
    x, y : NDArray[np.float64]
        Physical coordinates of every point of the map, in nm.

    Returns
    -------
    NDArray[np.float64]
        The peak position at every point, in cm-1.
    """
    return 1582.0 - 5.0 * x / X_EXTENT + 0.0 * y


def two_d_amplitude(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Height of the 2D peak, which drops over a round bilayer patch in the middle.

    Parameters
    ----------
    x, y : NDArray[np.float64]
        Physical coordinates of every point of the map, in nm.

    Returns
    -------
    NDArray[np.float64]
        The peak height at every point, in counts.
    """
    radius = np.hypot(x - X_EXTENT / 2, y - Y_EXTENT / 2)
    return np.where(radius < BILAYER_RADIUS_NM, 260.0, 900.0)


def background(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Photoluminescence background, growing towards the top of the map.

    Parameters
    ----------
    x, y : NDArray[np.float64]
        Physical coordinates of every point of the map, in nm.

    Returns
    -------
    NDArray[np.float64]
        The background level at every point, in counts.
    """
    return 120.0 + 80.0 * y / Y_EXTENT + 0.0 * x


def build_spec() -> MappingSpec:
    """Describe the synthetic mapping.

    Returns
    -------
    MappingSpec
        The spec of a 24 x 18 map of 1024-point spectra, with two peaks given as Raman shifts.
    """
    return MappingSpec(
        map=MapSpec(x_size=X_SIZE, y_size=Y_SIZE, x_step=STEP_NM, y_step=STEP_NM),
        # 575-630 nm covers roughly 1300-2900 cm-1 for a 532 nm excitation.
        spectral_axis=SpectralAxisSpec(size=1024, start=575.0, stop=630.0),
        peaks=[
            # A parameter can be a number, an array of the shape of the map, or a function of
            # the coordinates -- here, all three appear.
            PeakSpec(
                center=g_center,
                fwhm=16.0,
                amplitude=700.0,
                shape="lorentzian",
                units="raman_shift",
            ),
            PeakSpec(
                center=2685.0,
                fwhm=30.0,
                amplitude=two_d_amplitude,
                shape="lorentzian",
                units="raman_shift",
            ),
        ],
        baseline=BaselineSpec(offset=background, slope=-40.0),
        # Shot noise on the counts plus a little read noise, reproducible thanks to the seed.
        noise=NoiseSpec(poisson=True, sigma=4.0, seed=20260822),
        instrument=InstrumentSpec(
            laser_wavelength_nm=LASER_NM,
            laser_power_mw=1.2,
            exposure_time_s=0.5,
            accumulations=2,
            information="Synthetic graphene-like mapping",
        ),
    )


def band_intensity(
    mapping: Mapping, low: float, high: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Measure a band of every spectrum of the mapping.

    Parameters
    ----------
    mapping : Mapping
        The mapping to measure.
    low, high : float
        Limits of the band, in cm-1.

    Returns
    -------
    intensity : NDArray[np.float64]
        Height of the band above its own background, of the shape of the map.
    position : NDArray[np.float64]
        Center of mass of the band, in cm-1, of the shape of the map.

    Notes
    -----
    A center of mass stands in for a proper peak fit: it is a couple of lines, and unlike the
    position of the strongest point it does not jump from one pixel of the spectral axis to the
    next as the noise changes.
    """
    shift = mapping.get_spectral_axis("raman_shift")
    inside = (shift >= low) & (shift <= high)

    axis = shift[inside]
    band = mapping.get_map()[:, :, inside].astype(np.float64)

    # The median of the band is a decent stand-in for its background.
    above = np.clip(band - np.median(band, axis=-1, keepdims=True), 0.0, None)
    return above.max(axis=-1), (above * axis).sum(axis=-1) / above.sum(axis=-1)


def as_ascii(values: NDArray[np.float64]) -> str:
    """Draw a map as a block of shaded characters.

    Parameters
    ----------
    values : NDArray[np.float64]
        The map, of shape ``(y_size, x_size)``.

    Returns
    -------
    str
        One line per row of the map, brightest values as the densest characters. Rows are
        printed from the top of the scanned area downwards, following NanoFinder's convention
        that y grows upwards.
    """
    span = np.ptp(values)
    levels = (
        np.zeros_like(values, dtype=int)
        if span == 0
        else np.clip(
            ((values - values.min()) / span * (len(SHADES) - 1)).astype(int), 0, len(SHADES) - 1
        )
    )
    return "\n".join("    " + "".join(SHADES[level] * 2 for level in row) for row in levels[::-1])


def main(file: Path) -> None:
    """Write a synthetic SMD file, read it back, and print what it holds.

    Parameters
    ----------
    file : Path
        Path of the file to write.
    """
    spec = build_spec()
    create_smd(file, spec)

    mapping = load_smd(file)
    shift = mapping.get_spectral_axis("raman_shift")
    x_steps, y_steps, _ = mapping.map_steps

    print(f"\n{file}  ({file.stat().st_size / 1024:.0f} kB)")
    print(f"  grid            {x_steps} x {y_steps} points, {mapping.step_size[0]:.0f} nm apart")
    print(
        f"  spectra         {mapping.get_spectral_axis_len()} points, "
        f"{shift[0]:.0f}-{shift[-1]:.0f} cm-1"
    )
    print(f"  laser           {mapping.laser_wavelength} nm, {mapping.laser_power} mW")
    print(
        f"  acquisition     {mapping.get_exposure_time()} s x {mapping.get_accumulation_number()}"
    )
    print(f"  measured        {mapping.datetime}")

    g_intensity, g_position = band_intensity(mapping, *G_BAND)
    two_d_intensity, _ = band_intensity(mapping, *TWO_D_BAND)

    print(f"\n  G position, {g_position.min():.1f}-{g_position.max():.1f} cm-1, drifting along x:")
    print(as_ascii(g_position))
    print(
        f"\n  2D intensity, {two_d_intensity.min():.0f}-{two_d_intensity.max():.0f} counts, "
        "dim over the bilayer patch:"
    )
    print(as_ascii(two_d_intensity))

    ratio = g_intensity / two_d_intensity
    center = ratio[y_steps // 2, x_steps // 2]
    print(f"\n  G/2D ratio at the middle of the patch: {center:.2f}")
    print(f"  G/2D ratio at the corner:              {ratio[0, 0]:.2f}\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE)
