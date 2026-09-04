# Sample data

Small files used by the tests, the examples and the documentation.

| File | What it is |
| --- | --- |
| `smd/mapping_small.smd` | A real mapping trimmed down to a 4 x 3 grid of 8-point spectra |
| `s1d/kinetic_small.s1d` | A real kinetic series trimmed down to 10 spectra of 8 points |
| `mdt/Spectra.mdt` | Two individual spectra |
| `mdt/Spectra_and_2DMaps.mdt` | Two spectra and a 2-D map in a single file |

The values these files hold come from the instrument, so the tests can assert against real data.

The identifiers of the hardware they were recorded on have been replaced: every `AxisGuid` and
`DeviceGuid` of `mapping_small.smd` is now the nil UUID
(`{00000000-0000-0000-0000-000000000000}`), which is also what
[`nanofinderparser.write`](../src/nanofinderparser/write.py) writes for files created from
scratch. Only those 128 bytes of the XML header changed; the binary block, the spectral axis and
every other parameter are untouched.

`kinetic_small.s1d` was trimmed by writing the first 10 spectra of a real series back through
[`write_s1d`](../src/nanofinderparser/write.py), so the settings the parser models --- the
timings of the kinetic series, the laser, the detector temperature, the spectral axis --- and
the counts of the binary block are those of the instrument, while the fields the parser ignores
carry the defaults of that module. The path of the original file on the acquisition computer was
replaced by the name of the trimmed one.
