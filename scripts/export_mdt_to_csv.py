"""Export every MDT file of a folder to CSV.

Writes one CSV per spectrum and one per map, into an "exported" folder. The same thing can be
done from the command line with `nanofinderparser convert-mdt`; this script is a starting point
if you need to customise the output.

Run it from the project root:

    python scripts/export_mdt_to_csv.py
    python scripts/export_mdt_to_csv.py path/to/your/folder path/to/output
"""

import sys
from pathlib import Path

from nanofinderparser import load_mdt, load_mdt_images

# ruff: noqa: T201

SAMPLE_FOLDER = Path(__file__).parent.parent / "sample_data" / "mdt"


def export_folder(folder: Path, output: Path) -> None:
    """Export the spectra and maps of every MDT file of a folder to CSV.

    Parameters
    ----------
    folder : Path
        Folder holding the MDT files.
    output : Path
        Folder in which the CSV files will be written.
    """
    for file in sorted(folder.glob("*.mdt")):
        # Prefixing with the file stem keeps files from different sources apart.
        written = load_mdt(file).to_csv(output, filename=file.stem, spectral_units="raman_shift")
        written += load_mdt_images(file).to_csv(output, filename=file.stem)
        print(f"{file.name}: wrote {len(written)} CSV file(s)")


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_FOLDER
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("exported")  # noqa: PLR2004
    export_folder(folder, output)
