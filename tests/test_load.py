from pathlib import Path

import pytest
from nanofinderparser import load_smd, load_smd_folder
from nanofinderparser.models import Mapping


def test_load_smd(sample_smd_file):
    mapping = load_smd(sample_smd_file)
    assert isinstance(mapping, Mapping)
    # Add more specific assertions based on expected content of sample_smd_file


def test_load_smd_folder(tmp_path):
    # Create dummy SMD files
    for i in range(5):
        (tmp_path / f"test{i}.smd").touch()

    mappings = load_smd_folder(tmp_path)
    assert len(mappings) == 5
    assert all(isinstance(m, Mapping) for m in mappings)


def test_load_smd_folder_parallel(tmp_path):
    # Create dummy SMD files
    for i in range(15):
        (tmp_path / f"test{i}.smd").touch()

    mappings = load_smd_folder(tmp_path, parallel_threshold=10)
    assert len(mappings) == 15
    assert all(isinstance(m, Mapping) for m in mappings)


def test_load_smd_folder_sequential(tmp_path):
    # Create dummy SMD files
    for i in range(5):
        (tmp_path / f"test{i}.smd").touch()

    mappings = load_smd_folder(tmp_path, parallel_threshold=None)
    assert len(mappings) == 5
    assert all(isinstance(m, Mapping) for m in mappings)
