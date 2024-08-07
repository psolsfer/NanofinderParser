# NanoFinderParser Documentation

## Introduction

NanoFinderParser is a Python library for parsing SMD (Scanned Measurement Data) files produced by NanoFinder instruments. This library provides a set of tools to read, parse, and manipulate SMD file data.

## Installation

You can install NanoFinderParser using pip:

```bash
pip install nanofinderparser
```

## Basic Usage

### Loading an SMD File

To load an SMD file, use the `load_smd` function:

```python
from pathlib import Path
from nanofinderparser import load_smd

file_path = Path("path/to/your/smd/file.smd")
mapping_data = load_smd(file_path)
```

### Loading Multiple SMD Files from a Folder

To load multiple SMD files from a folder, use the `load_smd_folder` function:
```python
from pathlib import Path
from nanofinderparser import load_smd_folder

folder_path = Path("path/to/your/smd/files/folder")
mapping_data_list = load_smd_folder(folder_path)
```

The load_smd_folder function provides options for sequential, parallel, or adaptive loading based on the number of SMD files in the folder:

```python
# Force sequential loading
mapping_data_list = load_smd_folder(folder_path, parallel_threshold=None)

# Force parallel loading
mapping_data_list = load_smd_folder(folder_path, parallel_threshold=0)

# Custom threshold for parallel loading (e.g., use parallel if 20 or more files)
mapping_data_list = load_smd_folder(folder_path, parallel_threshold=20)
```

!!! note
    Parallel loading generally performs better when there are many files and/or the files are large.
    However, for a small number of files or very small files, sequential loading might be faster.
    The default threshold for parallel loading is 10 files.

### Accessing Parsed Data

Once you have loaded the SMD file, you can access various parts of the data through the `Mapping` object:

```python
# Access basic information
print(f"Exposure time: {mapping_data.get_exposure_time()}")
print(f"Laser power: {mapping_data.laser_power} mW")
print(f"Laser wavelength: {mapping_data.laser_wavelength_nm} nm")

print(f"Measurement date and time: {mapping_data.datetime}")



# Access the actual mapping data
print(f"Number of data points: {len(mapping_data.data)}")
print(f"Spectral axis: {mapping_data.get_spectral_axis()}")
```

!!! important
    It is recommended to create instances of this class using the `load_smd`
    function rather than instantiating it directly.

!!! note
    Currently, some methods that accept a 'channel' parameter that defaults to 'channel = 0'. At present, we don't have SMD files with multiple channels, so it's not yet clear how to handle them properly.
    Until we encounter multi-channel SMD files, it's recommended to keep using 'channel = 0' for all operations.

### Exporting Data

To export the loaded data, you can use the exporting methods of `Mapping`

```python
# Export as csv files
mapping_data.export_to_csv(path=Path("path/to/output/file"), spectral_units="raman_shift")

# Export to pandas DataFrames
data, map_coords = mapping_data.export_to_df(spectral_units="eV")
```

When exporting, it's possible to specify the units from ["nm", "cm-1", "eV", "raman_shift"].

When exporting to a .csv file, is possible to also export the coordinates of the mapping.

Note that NanoFinder's coordinates follow the convention of 'y' starting from the bottom of the mapping area.

## Advanced Usage

### Converting Spectral Units

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
