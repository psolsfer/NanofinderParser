#!/usr/bin/env python

"""Tests for `NanofinderParser` package."""

import re

from nanofinderparser.cli import app
from typer.testing import CliRunner


def test_command_line_interface() -> None:
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "nanofinderparser.app.main" in result.output
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert re.search(r"--help\s+Show this message and exit.", help_result.output) is not None
