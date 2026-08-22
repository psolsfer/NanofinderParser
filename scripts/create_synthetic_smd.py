"""Create a synthetic NanoFinder SMD file from the sample catalog and read it back.

Synthetic files are useful when a real mapping is too big to commit, too slow to load, or simply
does not exist yet: tests, examples and benchmarks can all be built from a spec instead. This
script writes one of the samples of :mod:`nanofinderparser.samples`, reads it back with the
normal parser, and prints what came out of it.

Run it from the project root:

    python scripts/create_synthetic_smd.py
    python scripts/create_synthetic_smd.py mos2
    python scripts/create_synthetic_smd.py graphene my_mapping.smd
"""

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nanofinderparser import create_smd, load_smd
from nanofinderparser.models import Mapping
from nanofinderparser.samples import SAMPLES, SampleName, sample_spec

# ruff: noqa: T201

SHADES = " .:-=+*#%@"


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


def report(mapping: Mapping, file: Path) -> None:
    """Print what a mapping holds.

    Parameters
    ----------
    mapping : Mapping
        The mapping just read back from the file.
    file : Path
        Path of the file it was read from.
    """
    axis = mapping.get_spectral_axis("nm")
    shift = mapping.get_spectral_axis("raman_shift")
    x_steps, y_steps, _ = mapping.map_steps
    cube = mapping.get_map().astype(np.float64)

    print(f"\n{file}  ({file.stat().st_size / 1024:.0f} kB)")
    print(f"  grid            {x_steps} x {y_steps} points, {mapping.step_size[0]:.0f} nm apart")
    print(
        f"  spectra         {mapping.get_spectral_axis_len()} points, "
        f"{axis[0]:.1f}-{axis[-1]:.1f} nm  ({shift[0]:.0f}-{shift[-1]:.0f} cm-1)"
    )
    print(f"  laser           {mapping.laser_wavelength} nm, {mapping.laser_power} mW")
    print(
        f"  acquisition     {mapping.get_exposure_time()} s x {mapping.get_accumulation_number()}"
    )
    print(f"  measured        {mapping.datetime}")

    brightest = cube.max(axis=-1)
    print(f"\n  Brightest count of each spectrum, {brightest.min():.0f}-{brightest.max():.0f}:")
    print(as_ascii(brightest))

    where = axis[cube.argmax(axis=-1)]
    print(f"\n  Where that maximum sits, {where.min():.1f}-{where.max():.1f} nm:")
    print(as_ascii(where))
    print()


def main() -> None:
    """Write a synthetic SMD file, read it back, and print what it holds."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "sample",
        nargs="?",
        default="graphene",
        choices=list(SAMPLES),
        help="\n".join(f"{info.name}: {info.description}" for info in SAMPLES.values()),
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        default=None,
        help="Path of the file to write, by default '<sample>.smd'.",
    )
    arguments = parser.parse_args()

    name: SampleName = arguments.sample
    file: Path = arguments.file or Path(f"{name}.smd")

    create_smd(file, sample_spec(name))
    print(f"\n{SAMPLES[name].title} -- {SAMPLES[name].description}")
    report(load_smd(file), file)


if __name__ == "__main__":
    main()
