from pathlib import Path
from typing import Any

import pytest
from nanofinderparser import load_smd, load_smd_folder
from nanofinderparser.models import Mapping


def test_load_smd(sample_smd_file: Any) -> None:
    mapping = load_smd(sample_smd_file)
    assert isinstance(mapping, Mapping)
    # Add more specific assertions based on expected content of sample_smd_file


def test_load_smd_folder(tmp_path: Path) -> None:
    # Create dummy SMD files
    for i in range(5):
        (tmp_path / f"test{i}.smd").touch()

    mappings = []
    for mp in load_smd_folder(tmp_path):
        mappings.append(mp)
    assert len(mappings) == 5
    assert all(isinstance(m, Mapping) for m in mappings)
