# NanoFinderParser Documentation

## Introduction

NanoFinderParser is a Python library for parsing SMD (Scanned Measurement Data) files produced by NanoFinder instruments. This library provides a set of tools to read, parse, and manipulate SMD file data.

## Installation

You can install NanoFinderParser using pip:

```bash
pip install nanofinderparser
```

## Basic usage

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
print(f"Laser wavelength: {mapping.laser_wavelength_nm} nm")
print(f"Measurement date and time: {mapping.datetime}")

# Access the actual mapping data
data = mapping.data
# and spectral axis (in the specified units)
spectral_axis = mapping.get_spectral_axis("eV")
print(f"Number of data points: {len(data)}")
print(f"Spectral axis (eV): {spectral_axis}")
```

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
mapping.to_csv(path=Path("path/to/output/file"), spectral_units="raman_shift", save_mapcoords="separated")
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

    * **Columns**: 'x' and 'y' (and potentially 'z' for 3D mappings).
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

* [Models](api/models.md)
* [Parsers](api/parsers.md)
* [Units](api/units.md)
