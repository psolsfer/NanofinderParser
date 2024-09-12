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

!!! important
    It is recommended to create instances of this class using the `load_smd`
    function rather than instantiating it directly.

!!! note
    Currently, some methods that accept a 'channel' parameter that defaults to 'channel = 0'. At present, we don't have SMD files with multiple channels, so it's not yet clear how to handle them properly.
    Until we encounter multi-channel SMD files, it's recommended to keep using 'channel = 0' for all operations.

### Exporting data

To export the loaded data, you can use the exporting methods of `Mapping`

```python
# Export the data as csv files in the specified units
mapping.to_csv(path=Path("path/to/output/file"), spectral_units="raman_shift")

# Export the data to pandas DataFrames
data, map_coords = mapping_data.to_df(spectral_units="eV")
```

When exporting, it's possible to specify the units from ["nm", "cm-1", "eV", "raman_shift"].

When exporting to a .csv file, is possible to also export the coordinates of the mapping with the 'save_mapcoords' argument. By default, the coordinates will be saved to the same .csv file.

Note that NanoFinder's coordinates follow the convention of 'y' starting from the bottom of the mapping area.

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

- [Models](api/models.md)
- [Parsers](api/parsers.md)
- [Units](api/units.md)
