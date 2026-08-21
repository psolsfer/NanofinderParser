# NanoFinderParser Documentation

## Introduction

NanoFinderParser is a Python library for parsing the data files produced by NanoFinder instruments. This library provides a set of tools to read, parse, and manipulate them.

There are two kinds of file, and each has its own loader:

| File | Contents | Loader |
| ---- | -------- | ------ |
| `.smd` | A mapping: one spectrum per point of a spatial scan | `load_smd` |
| `.mdt` | Individual spectra, and 2-D maps | `load_mdt`, `load_mdt_images` |

## Installation

You can install NanoFinderParser using pip:

```bash
pip install nanofinderparser
```

## Mapping files (`.smd`)

### Loading an SMD file

To load an SMD file, use the `load_smd` function:

```python
from pathlib import Path
from nanofinderparser import load_smd

file_path = Path("path/to/your/smd/file.smd")
mapping = load_smd(file_path)
```

### Loading multiple SMD files from a folder

To load multiple SMD files from a folder, use the `load_smd_folder` function:

```python
from pathlib import Path
from nanofinderparser import load_smd_folder

folder_path = Path("path/to/your/smd/files/folder")
for mapping in load_smd_folder(folder_path):
    # The loaded file is stored in 'mapping', and it's possible to access its properties
    data = mapping.data
```

### Accessing parsed data

Once you have loaded the SMD file, you can access various parts of the data through the `Mapping` object:

```python
# Access basic information of the mapping
print(f"Exposure time: {mapping.get_exposure_time()}")
print(f"Laser power: {mapping.laser_power} mW")
print(f"Laser wavelength: {mapping.laser_wavelength} nm")
print(f"Measurement date and time: {mapping.datetime}")

# Access the actual mapping data
data = mapping.data
# and spectral axis (in the specified units)
spectral_axis = mapping.get_spectral_axis("eV")
print(f"Number of data points: {len(data)}")
print(f"Spectral axis (eV): {spectral_axis}")
```

`mapping.data` is the flat array exactly as stored in the file. Two reshaped views of it are
usually more convenient:

```python
spectra = mapping.get_spectra()  # (n_spectra, spectral_len), in acquisition order
cube = mapping.get_map()  # (slow_axis, fast_axis, spectral_len), the spatial map
```

Both are views, so they cost nothing and share memory with `mapping.data`.

!!! tip
    It is recommended to create instances of the `Mapping` class using the `load_smd` function rather than instantiating it directly.

!!! note
    Currently, some methods that accept a 'channel' parameter that defaults to 'channel = 0'. At present, it is not possible to handle multi-channel SMD files. Until this is solved, it's recommended to keep using 'channel = 0' for all operations.

### Exporting data

To export the loaded data, you can use the exporting methods of `Mapping`.

!!! tip "Exporting units"
    When exporting, you can specify the spectral units from `["nm", "cm-1", "eV", or "raman_shift"]`.

#### Exporting to CSV

Export the data as CSV files in the specified units:

```python
mapping.to_csv(path=Path("path/to/output/file"), spectral_units="raman_shift")
```

When exporting to a CSV file, it is possible to control how the mapping coordinates are saved using the `save_mapcoords` argument:

* `combined` (default): Coordinates are saved in the same file as the spectral data.
* `separated`: Coordinates are saved in a separate file.
* `no`: Coordinates are not saved.

Example:

```python
mapping.to_csv(
    path=Path("path/to/output/file"), spectral_units="raman_shift", save_mapcoords="separated"
)
```

The data is composed of a row for each of the spectra, with the top row being the spectral axis (e.g., Raman shift in cm-1, or energy in eV).

#### Exporting to pandas DataFrames

!!! info "Exporting Data"
    You can export the data to pandas DataFrames using `to_df` method of `Mapping`:

    ```python
    data, map_coords = mapping_data.to_df(spectral_units="eV")
    ```

The returned objects have the following structure:

1. `data`: A DataFrame containing the spectral data, with the spectral axis as the header.

    * **Columns**: Represent the spectral axis (e.g., Raman shift in cm⁻¹, wavelength in nm, or energy in eV).
    * **Rows**: Each row corresponds to a single spectrum from a specific point in the mapping.
    * **Index**: By default, a multi index with (x, y) mapping coordinates of each spectrum.

2. `map_coords`: A DataFrame containing the mapping coordinates.

    * **Columns**: 'x' and 'y' (and potentially 'z' for 3D mappings), as `pint`-aware columns carrying the stage's physical units (e.g. nm) when available.
    * **Rows**: Correspond to the spectra in the same order as in the data DataFrame.

!!! note "Customizing the Index"
    By default, the index of the `data` DataFrame will be the mapping coordinates (x, y). You can change this behavior with the `index` argument:

    ```python
    data, map_coords = mapping.to_df(spectral_units="eV", index=False)
    ```

    This will reset the index to a simple numeric index.


#### Notes

* NanoFinder's coordinates follow the convention of 'y' starting from the bottom of the mapping area.
* The channel parameter in both methods allows you to specify which channel to export if you're working with multi-channel data. By default, it uses channel 0.

### CLI usage

The NanofinderParser package provides a command-line interface (CLI) for easy conversion of SMD files to CSV format and for displaying information about SMD files.

#### Converting SMD files to CSV

You can convert SMD files to CSV directly from the command line:

```shell
nanofinderparser convert input_file.smd [output_folder]
```

* If the output folder is not specified, the CSV file will be saved in the same directory as the input file.
* If the input is a directory, all SMD files in that directory will be converted.

Options:

* --units: Specify the units for the spectral axis (default: raman_shift)
* --save-mapcoords: Specify how to save mapping coordinates (default: combined)

Example:

```shell
nanofinderparser convert mapping_file.smd output_folder --units nm --save-mapcoords separated
```

#### Displaying SMD file information

To display information about an SMD file:

```shell
nanofinderparser info mapping_file.smd
```

This command will show details such as the laser wavelength and power, exposure time, map and step size, ...

## Individual spectra (`.mdt`)

When you measure single points instead of a mapping, NanoFinder saves the result as an MDT file. One file usually holds **several** spectra, each with its own title, so `load_mdt` returns a collection:

```python
from pathlib import Path
from nanofinderparser import load_mdt

spectra = load_mdt(Path("path/to/your/file.mdt"))

print(spectra.titles)  # ['Spectrum_1', 'Spectrum_2']
print(len(spectra))  # 2
```

You can get a spectrum by position or by title, iterate over the collection, or slice it:

```python
first = spectra[0]
named = spectra["Spectrum_2"]

for spectrum in spectra:
    print(spectrum.title, spectrum.datetime)
```

### Working with a spectrum

Each `Spectrum` carries its own spectral axis and its intensities as numpy arrays, plus the metadata of the measurement:

```python
spectrum = spectra["Spectrum_1"]

print(f"Title: {spectrum.title}")
print(f"Laser wavelength: {spectrum.laser_wavelength} nm")
print(f"Measured at: {spectrum.datetime}")
print(f"Points: {spectrum.spectral_axis_len}")

# The axis as stored (NanoFinder writes nm), and converted to the units you work in
wavelength = spectrum.spectral_axis
raman_shift = spectrum.get_spectral_axis("raman_shift")
intensity = spectrum.data
```

!!! warning "Spectra in the same file may not share a spectral axis"
    Each spectrum is an independent measurement, and may have been recorded at a different grating position. Always take the axis from the spectrum you are plotting, rather than reusing the one from another.

### Exporting spectra

A single spectrum exports to a DataFrame indexed by the spectral axis:

```python
df = spectrum.to_df(spectral_units="raman_shift")
```

The whole collection exports too. By default `to_csv` writes **one file per spectrum**, which is the safe option when the axes differ:

```python
spectra.to_csv(path=Path("output"), spectral_units="raman_shift")
```

Use `combined=True` to put everything in a single file, with one column per spectrum. If the spectra do not share an axis, the columns are aligned on the union of the axes, leaving empty cells, and a warning is emitted:

```python
spectra.to_csv(path=Path("output"), filename="all", combined=True)
df = spectra.to_df(spectral_units="raman_shift")
```

!!! note "Layout"
    This is transposed with respect to `Mapping.to_df`: spectra go in **columns**, not rows. In a mapping every row is tied to a map coordinate, whereas here each spectrum stands on its own.

### 2-D maps

MDT files can also hold maps: either measured directly (a PL intensity map, say) or produced by fitting each spectrum of a mapping, such as the position or FWHM of a peak. Those are read with `load_mdt_images`:

```python
from nanofinderparser import load_mdt_images

images = load_mdt_images(Path("path/to/your/file.mdt"))

for image in images:
    print(image.title, image.shape, image.value_unit)

peak_map = images["G Peak position (Lorentz)"]
values = peak_map.values  # 2-D array, shape (y, x)
df = peak_map.to_df()  # indexed by the physical coordinates
images.to_csv(path=Path("output"))  # one CSV per map
```

A file may hold both kinds. `load_mdt` reads the spectra, `load_mdt_images` the maps, and neither loses anything the other reads.

Each of those reads and decodes the whole file, so if you want both, ask for both at once:

```python
from nanofinderparser import load_mdt_file

spectra, images = load_mdt_file(Path("path/to/your/file.mdt"))
```

!!! note "Map values are quantized"
    Maps are stored with 65535 levels spanning their own range of values, so the resolution is `(max - min) / 65535`. That is finer than one count for most maps, but not for those covering a very wide range.

### CLI usage

```shell
# Convert the spectra of an MDT file (or of every MDT file in a folder) to CSV
nanofinderparser convert-mdt input_file.mdt [output_folder] --units nm

# Write all the spectra of each file to a single CSV, and export the maps too
nanofinderparser convert-mdt input_file.mdt output_folder --combined --maps

# List what a file contains
nanofinderparser info-mdt input_file.mdt
```

### Sample files and example scripts

The repository ships a few real files in `sample_data/` — MDT files in `sample_data/mdt/`, and a mapping trimmed to a 4 x 3 grid in `sample_data/smd/` — and short example scripts in `scripts/` that run against them:

```shell
python scripts/explore_mdt.py          # list the contents of a file
python scripts/plot_mdt_spectra.py     # plot the spectra (needs matplotlib)
python scripts/export_mdt_to_csv.py    # export a whole folder to CSV
```

Each accepts a path, so you can point them at your own files.

## Advanced usage

### Converting spectral units

You can convert spectral data between different units using the `convert_spectral_units` function:

```python
from nanofinderparser.units import convert_spectral_units

# Convert wavelength (nm) to wavenumber (cm-1)
wavelength_nm = 532.0
wavenumber_cm1 = convert_spectral_units(wavelength_nm, "nm", "cm_1")
print(f"{wavelength_nm} nm is equal to {wavenumber_cm1:.2f} cm-1")

# Convert an array of values
import numpy as np

wavelengths = np.array([500, 550, 600])
energies_ev = convert_spectral_units(wavelengths, "nm", "eV")
print(f"Energies: {energies_ev}")
```

## API Reference

For detailed information about classes and functions, please refer to the API documentation:

* [Load](../api/load.md)
* [Models](../api/models.md)
* [Parsers](../api/parsers.md)
* [Units](../api/units.md)

The internal layout of both file formats is described in
[File formats](../concepts/file-formats.md).
