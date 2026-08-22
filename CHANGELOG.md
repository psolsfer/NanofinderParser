## v0.7.0 (2026-08-22)

### Feat

- **synthetic**: add a catalog of ready-made synthetic samples
- add writing of smd files

## v0.6.0 (2026-08-21)

### Feat

- **load**: add load_mdt_file to read spectra and maps in one pass
- **load**: check the SMD data block against its header
- **models**: add Mapping.source and a public get_map()
- add support for mdt files
- **Mapping**: return pint-unit-aware mapping coordinates from to_df

### Fix

- **units**: raise the intended error for an invalid unit name

### Refactor

- **utils**: drop the imports left by the removed helper
- remove unused code

### Perf

- **models**: reorder mapping rows by position instead of by label
- **units**: reuse one pint registry across conversions
- **parsers**: read the SMD binary block with numpy

## v0.5.1 (2026-08-20)

## v0.5.0 (2026-08-20)

### Feat

- improve SMD XML parsing coverage and fix axis metadata handling

### Fix

- correct z-axis step_size bug and refactor Mapping data storage
- fix bug in _get_data_to_map

## v0.4.1 (2025-06-25)

### Feat

- **load.py**: load_smd_folder can also yield path of loaded smd files

## v0.4.0 (2025-05-20)

## v0.3.7 (2025-01-22)

## v0.3.6 (2025-01-22)

### Fix

- **nanofinderparser.models.Axis**: fix parsing axis inversion for new nanofinder versions

## v0.3.5 (2024-11-08)

## v0.3.4 (2024-10-15)

### Fix

- **models.Mapping**: Fix the index of mapcoords outputted by Mapping.to_df

## v0.3.3 (2024-10-15)

### Fix

- **models.Mapping**: fix bug with the indexes of the dataframes exported by .to_df

## v0.3.2 (2024-10-02)

### Refactor

- **units.py**: improve the static typing of the unit conversion

## v0.3.1 (2024-10-02)

### Fix

- **units.py**: fix bug in the conversion to and from "cm-1"

## v0.3.0 (2024-10-02)

### Feat

- **Mapping.to_df**: add control of the index in the created df

## v0.2.2 (2024-09-12)

### Feat

- **cli.py**: improve the cli

### Fix

- **models**: fix parsing of channel info

## v0.2.1 (2024-09-12)

### Feat

- **Mapping**: change name of methods to export and specify units in get_spectral_axis
