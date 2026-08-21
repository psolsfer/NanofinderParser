"""Explore the contents of a NanoFinder MDT file.

MDT files are containers: a single file can hold several individual spectra, several 2-D maps,
or a mix of both. This script prints what is inside one, which is usually the first thing you
want to know.

Run it from the project root:

    python scripts/explore_mdt.py
    python scripts/explore_mdt.py path/to/your/file.mdt
"""

import sys
from pathlib import Path

from nanofinderparser import load_mdt, load_mdt_images

SAMPLE = Path(__file__).parent.parent / "sample_data" / "mdt" / "Spectra_and_2DMaps.mdt"


def explore(file: Path) -> None:
    """Print a summary of the spectra and maps held by an MDT file.

    Parameters
    ----------
    file : Path
        The MDT file to explore.
    """
    spectra = load_mdt(file)
    images = load_mdt_images(file)

    print(f"\n{file.name}: {len(spectra)} spectra, {len(images)} maps")

    for spectrum in spectra:
        axis = spectrum.spectral_axis
        print(
            f"  spectrum  {spectrum.title:<32} {spectrum.spectral_axis_len:>5} points  "
            f"{axis[0]:.1f}-{axis[-1]:.1f} {spectrum.spectral_axis_unit}  "
            f"laser {spectrum.laser_wavelength} nm  {spectrum.datetime}"
        )

    for image in images:
        y_size, x_size = image.shape
        print(
            f"  map       {image.title:<32} {x_size:>3} x {y_size:<3}  "
            f"{image.values.min():.4g} to {image.values.max():.4g} {image.value_unit}"
        )


if __name__ == "__main__":
    explore(Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE)
