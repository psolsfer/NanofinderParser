# File formats

NanoFinder writes two file formats, and this page describes how both are laid out on disk. It
is meant as a reference for anyone maintaining the parsers, or trying to make sense of a file
that does not load.

!!! warning "No official specification"
    Neither format is publicly documented. Everything below was obtained by reverse engineering
    real files, cross-checked against the `.mdt` reader of
    [Gwyddion](http://gwyddion.net/) and against
    [NTMDTRead.py](https://github.com/psolsfer/NTMDTRead.py). Fields not used by the parsers are
    described only where their size matters for finding the next field.

| | `.smd` | `.mdt` |
| --- | --- | --- |
| Written for | A mapping: one spectrum per point of a spatial scan | Individual spectra, and 2-D scalar maps |
| Structure | One XML header + one binary block | A sequence of self-contained binary frames |
| Holds | Exactly one measurement | Any number of measurements, of both kinds |
| Spectral data | `float32`, full precision | `float32`, full precision |
| Map data | Full spectra | A single `int16` value per point |
| Stage coordinates | Yes, from the scan parameters | Only for maps; spectra carry none |
| Origin | NanoFinder's own format | NT-MDT's format, shared with NTEGRA and Gwyddion |
| Parsed by | `load_smd` | `load_mdt`, `load_mdt_images` |

## SMD — mapping files

An SMD file is a text header followed by a raw binary block:

```text
┌──────────────────────────────────────────────┐
│ <?xml version="1.0"?>                        │
│ <SCANDATA>  … all the metadata …             │  ASCII text
│ </SCANDATA>                                  │
├──────────────────────────────────────────────┤
│ 0D 0A                                        │  end of the last XML line
├──────────────────────────────────────────────┤
│ float32 float32 float32 …                    │  DataBlockSizeBytes bytes
└──────────────────────────────────────────────┘
```

There is no length prefix and no offset table: the binary block simply starts on the byte after
the line holding `</SCANDATA>`. This is why `read_xml_part` reads the file line by line and
returns the file position it stopped at, rather than parsing the whole file as XML.

!!! note "Line endings"
    NanoFinder writes each line with `\r\r\n` — a stray carriage return before the usual CRLF,
    a fingerprint of Visual Basic's `Print #`. Reading the file in binary mode therefore yields
    one element per line and no blank lines, which is what the line-by-line scan relies on.

### The XML header

Only the elements the parsers use are listed. NanoFinder writes a good deal more, including
scan-mode flags, axis limits, and DAC calibration bounds.

`SCANDATA`

| Element | Maps to | Notes |
| --- | --- | --- |
| `Vendor`, `Version` | `Mapping.vendor`, `.version` | `NNFinder`, `1` |
| `ScannedFrameParameters` | `ScannedFrameParameters` | everything below |

`SCANDATA/ScannedFrameParameters`

| Element | Maps to | Notes |
| --- | --- | --- |
| `ScanRepeatNumber` | `.scan_repeat_number` | |
| `FrameHeader` | `FrameHeader` | when and on what |
| `FrameOptions` | `FrameOptions` | excitation and spectrometer |
| `Stage3DParameters` | `Stage3DParameters` | the scan grid |
| `DataCalibration` | `DataCalibration` | the detector channels |
| `OriginalFileName` | `.original_file_name` | full path on the acquisition PC |
| `DataBlockSizeBytes` | `.data_block_size_bytes` | size of the binary block |

`FrameHeader` carries `Date` (`YYYY/MM/DD`), `Time` (`HH:MM:SS`), `Information`, `SystemName`,
`PositioningSysName`, `DetectionSysName` and `ScannedDataName` (`Mapping` for a mapping).

`FrameOptions` carries the excitation and spectrometer settings actually needed to interpret the
data: `OmuLaserWLnm` (excitation wavelength, nm), `OmuCurPower` (mW), `OmuGratingGroove`
(grooves/mm), `OmuCentralWaveLengthNM`, and `OmuPinHoleSize` (µm).

#### The scan grid

`Stage3DParameters` gives the number of points per axis in `AxisSizeX`, `AxisSizeY` and
`AxisSizeZ`, and the calibration of each axis under `StageAxesDimentions/AxisX|AxisY|AxisZ`:

| Element | Meaning |
| --- | --- |
| `AxisIsInUse` | whether the axis took part in the scan |
| `AxisIsInversed` | whether the axis direction is reversed |
| `AxisIsSlow` | whether the axis is the slow one of the raster |
| `AxisName`, `AxisUnitName` | e.g. `X` and `nm` |
| `AxisCountStart`, `AxisCountStep` | raw DAC counts (0 – 65535) |
| `AxisBiasFloat`, `AxisScaleFloat` | offset and counts → physical units |

Positions are stored as raw 16-bit DAC counts, not as physical coordinates, so they have to be
converted:

```text
step_size      = AxisCountStep  × AxisScaleFloat
start_position = AxisBiasFloat  + AxisCountStart × AxisScaleFloat
```

!!! note "Visual Basic booleans"
    `AxisIsInUse`, `AxisIsInversed` and `AxisIsSlow` follow the VB6 convention: `-1` is true and
    `0` is false. In practice both in-plane axes report `AxisIsSlow = 0`, so the parser falls
    back to NanoFinder's default raster — x fast, y slow.

#### The detector channels

`DataCalibration/DataDimentions` holds one `Channel0`, `Channel1`, … element per detector
channel; only single-channel files have been seen so far.

| Element | Meaning |
| --- | --- |
| `DeviceGuid`, `DeviceName` | e.g. `Andor CCD` |
| `DataChannelName`, `DataChannelUnit` | e.g. `Photons`, `Counts` |
| `ChannelSize` | points per spectrum |
| `ChannelAxisName`, `ChannelAxisUnit` | e.g. `Wavelength`, `nm` |
| `ChannelAxisLaserWl` | excitation wavelength used for the calibration |
| `ChannelAxisArray` | the spectral axis: `ChannelSize` space-separated decimals |
| `SeriesSize` | acquisitions per spatial point; only `1` is supported |
| `ChannelInfo` | free-text `Item0`, `Item1`, … |

The spectral axis is stored explicitly, value by value, rather than as a polynomial calibration —
which is why the header of a 1024-point file is already ~26 kB.

`ChannelInfo` is a list of human-readable strings rather than structured elements, and
`Channel.parse_channel_info` picks the useful ones apart:

```xml
<Item0>Head model = DV420</Item0>
<Item3>Central Pixel = 510</Item3>
<Item6>Acquisition mode: Accomulate. Number in Accumulation = 1</Item6>
<Item8>Exposure time, [sec] = 0.5000 Cycle time [sec] = 0.5730</Item8>
<Item10>Temperature [grad C] = -60.00</Item10>
```

Note that some items use `=` and others `:`, that the units are spelled inside the key, and that
`Accumulate` is misspelled `Accomulate` — all of which the parser has to accommodate.

!!! warning "NanoFinder's own typos are part of the format"
    `StageAxesDimentions`, `DataDimentions`, `AxisFloatPrecition`, `DerectionMode` and
    `Accomulate` are spelled that way *in the files*. They must not be "fixed" in the models.

### The binary block

Little-endian `float32`, one value per spectral point, with no padding or separators:

```text
number of values = AxisSizeX × AxisSizeY × AxisSizeZ × SeriesSize × ChannelSize
DataBlockSizeBytes = 4 × number of values
```

The values run in acquisition order: each spectrum is contiguous, spectra advance along x first,
then y, then z. Reshaping to `(y, x, spectral)` therefore gives the spatial map directly, and is
what `Mapping.get_map` does.

A 50 × 50 map of 1024-point spectra gives 2500 spectra, 2 560 000 values and
`DataBlockSizeBytes = 10 240 000` — the whole reason SMD files are tens of megabytes.

!!! note "y grows upwards"
    NanoFinder's convention is that y starts at the *bottom* of the mapping area.
    `Mapping.to_df` re-orders the rows accordingly, so the exported order is not the
    acquisition order.

## MDT — spectra and 2-D maps

An MDT file is a container: a 33-byte file header followed by frames laid end to end, each one
declaring its own length. A frame is either a spectrum or a map, and a single file can hold both,
in any order — the sample file `Spectra_and_2DMaps.mdt` stores a spectrum, then a map recorded
two years earlier, then another spectrum.

```text
┌──────────────────────────────────────────────┐
│ file header                          33 bytes│
├──────────────────────────────────────────────┤
│ frame 0   header (22 B) + body               │
├──────────────────────────────────────────────┤
│ frame 1   header (22 B) + body               │
├──────────────────────────────────────────────┤
│ …                                            │
└──────────────────────────────────────────────┘
```

### File header

| Offset | Size | Type | Meaning |
| ---: | ---: | --- | --- |
| 0 | 4 | bytes | signature `01 B0 93 FF` |
| 4 | 4 | `uint32` | file size, excluding this header |
| 8 | 4 | — | reserved, zero |
| 12 | 2 | `uint16` | index of the **last** frame, i.e. frame count − 1 |
| 14 | 2 | — | reserved (`00 10` in NanoFinder files) |
| 16 | 17 | `char[]` | `{c} 1998, NT-MDT` and its NUL terminator |

### Frame header

Fixed, 22 bytes, little-endian throughout (`struct` format `<IHBBHHHHHHH`):

| Offset | Size | Type | Meaning |
| ---: | ---: | --- | --- |
| 0 | 4 | `uint32` | frame size, **including** this header |
| 4 | 2 | `uint16` | frame type — `0` = 2-D map, `2` = spectrum |
| 6 | 1 | `uint8` | version, major |
| 7 | 1 | `uint8` | version, minor |
| 8 | 12 | 6 × `uint16` | year, month, day, hour, minute, second |
| 20 | 2 | `uint16` | `var_size`, the size of the frame-variables block |

The frame size is the only way to find the next frame, so a single corrupt frame makes everything
after it unreadable.

### Frame body

Offsets below are relative to the end of the 22-byte header. `var_size` is 442 bytes in every
NanoFinder frame seen so far.

| Offset | Size | Contents |
| ---: | ---: | --- |
| 0 | 10 | x axis scale |
| 10 | 10 | y axis scale |
| 20 | 10 | z axis scale |
| 32 | 4 | value counts, 2 × `uint16` — **often left at zero** |
| … | | rest of the frame variables |
| `var_size` | 8 | data header; the value counts appear again at `var_size + 2` |
| `var_size + 8` | | the data array |
| after the data | 4 + n | title: `uint32` length, then cp1252 bytes |
| | 4 + n | comment: `uint32` length, then UTF-16LE bytes |
| | rest | trailer, not parsed |

An **axis scale** record is 10 bytes (`<ffh`): `float32` offset, `float32` step, `int16` unit
code. The unit codes are shared with NT-MDT's own software and are listed in `MdtUnit`; the ones
NanoFinder writes are `-2` (micrometer), `-1` (nanometer), `3` (dimensionless) and `13` (counts).

!!! warning "The value counts are stored twice, and one copy is often empty"
    Both the frame variables (offset 32) and the data header (`var_size + 2`) declare how many
    values the frame stores. NanoFinder does not reliably fill in the first copy: in
    `Spectra.mdt` both frames leave it at `(0, 0)`, while in `Spectra_and_2DMaps.mdt` the last
    frame fills it in. `_read_mdt_value_counts` therefore prefers the copy next to the data,
    falls back to the other one, and refuses a frame whose two copies disagree.

### Spectrum frames (type 2)

The two counts are `(point_count, array_count)`, and NanoFinder always writes `array_count = 2`.
The data block is `array_count × point_count` `float32` values:

* row 0 — the spectral axis, in the units of the **x** scale (nanometer)
* row 1 — the intensities, in the units of the **z** scale (counts)

Both rows are already in physical units, so the `offset` and `step` of the axis scales are *not*
applied. A 1024-point spectrum gives a frame of 9048 bytes.

### Map frames (type 0)

The two counts are `(x_size, y_size)`. The data block is `x_size × y_size` **signed `int16`**
values, row-major with x as the fast axis. Physical values are recovered from the z scale:

```text
value = z_scale.offset + raw × z_scale.step
```

The x and y scales give the start position and the step of the two spatial axes, so a map carries
its own coordinates — unlike the spectra in the same file.

!!! warning "Map values are quantized"
    Storing a map as `int16` gives it 65 536 levels spanning its own range, so its resolution is
    `(max − min) / 65535`. That is finer than one count for most maps, but not for one covering a
    wide range: a map running from 1 × 10⁵ to 2 × 10⁵ counts resolves steps of about 1.7 counts,
    and the original values cannot be recovered any better than that. Fit parameter maps exported
    from NanoFinder are lossy for the same reason.

### The frame comment

The comment is an XML document stored as UTF-16LE, and is where the excitation wavelength hides —
it is not part of any binary field:

```xml
<?xml version="1.0" encoding="UTF-16"?>
<FrameComment>
	<TextComment></TextComment>
	<Parameters><SWLaserWL>532</SWLaserWL></Parameters>
</FrameComment>
```

`Parameters` is empty (`<Parameters/>`) on map frames, which is why an `Image` has no
`laser_wavelength` while a `Spectrum` does.

### What is not parsed

Every frame ends with a trailer that the parser skips: about 50 bytes on spectrum frames, and
around 3.2 kB on map frames. The latter holds the display settings — including the color palette
— and a block of scan parameters. Reading it is only needed to reproduce NanoFinder's own
rendering of a map, never to recover the data.

Frames whose type is neither `0` nor `2` are skipped with a warning rather than treated as an
error, so an unfamiliar frame in the middle of a file does not make the rest unreadable.

## Consequences for the parsers

A few practical points follow from the above, and are worth keeping in mind when changing the
code:

* **An SMD file has no index.** The binary block is found by scanning text, so anything that
  changes how the header is read (encoding, line endings, a stray blank line) breaks the data
  read as well.
* **`DataBlockSizeBytes` is a checksum in disguise.** It should agree with
  `4 × AxisSizeX × AxisSizeY × AxisSizeZ × SeriesSize × ChannelSize`; a mismatch means the file is
  truncated or holds something the parser does not model yet.
* **MDT frames are only reachable in order.** There is no seeking to frame *n*, which is why
  `read_mdt_frames` reads the whole file at once.
* **MDT spectra have no coordinates.** Nothing in a spectrum frame says where on the sample it
  was taken; if that matters, it has to come from the title or the free-text comment.
