"""Plot the spectra of a NanoFinder MDT file, in Raman shift.

Shows the typical workflow: load the file, convert the spectral axis to the units you work in,
and use the data. Requires matplotlib, which is not a dependency of the library:

    uv pip install matplotlib

Run it from the project root:

    python scripts/plot_mdt_spectra.py
    python scripts/plot_mdt_spectra.py path/to/your/file.mdt
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from nanofinderparser import load_mdt

SAMPLE = Path(__file__).parent.parent / "sample_data" / "mdt" / "Spectra_and_2DMaps.mdt"


def plot_spectra(file: Path) -> None:
    """Plot every spectrum of an MDT file against the Raman shift.

    Parameters
    ----------
    file : Path
        The MDT file to plot.
    """
    spectra = load_mdt(file)
    if not spectra:
        print(f"{file.name} holds no individual spectra.")
        return

    _, ax = plt.subplots(figsize=(9, 5))
    for spectrum in spectra:
        # Each spectrum may have its own axis, so convert them one by one.
        ax.plot(spectrum.get_spectral_axis("raman_shift"), spectrum.data, label=spectrum.title)

    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel(f"Intensity ({spectra[0].data_unit})")
    ax.set_title(file.name)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_spectra(Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE)
