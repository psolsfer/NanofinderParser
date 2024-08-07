#!/usr/bin/env python

"""Tests for `NanofinderParser` package."""

import pytest


# from nanofinderparser import nanofinderparser
@pytest.fixture
def response() -> None:
    """Sample pytest fixture."""
    # import requests
    # return requests.get("https://github.com/psolsfer/nanofinderparser")"""


def test_content(response) -> None:
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert "GitHub" in BeautifulSoup(response.content).title.string
