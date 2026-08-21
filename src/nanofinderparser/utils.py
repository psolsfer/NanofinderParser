"""Utilities."""

import operator
from enum import StrEnum
from functools import reduce
from typing import Any


class SaveMapCoords(StrEnum):
    """Enumeration for specifying how mapping coordinates should be saved.

    Attributes
    ----------
        no (str): Do not save mapping coordinates.
        combined (str): Save mapping coordinates in the same file as the data.
        separated (str): Save mapping coordinates in a separate file.
    """

    no = "no"
    combined = "combined"
    separated = "separated"


def validate_savemapcoords(savemapcoords: SaveMapCoords | str | Any) -> SaveMapCoords:
    """Convert string to SaveMapCoords enum if necessary and validate the input.

    Parameters
    ----------
    savemapcoords : SaveMapCoords or str
        The savemapcoords to check and potentially convert.

    Returns
    -------
    Units
        The validated SaveMapCoords enum value.

    Raises
    ------
    ValueError
        If the input is not a valid SaveMapCoords enum value or string representation.
    """
    if isinstance(savemapcoords, str):
        try:
            return SaveMapCoords(savemapcoords.lower())
        except ValueError as err:
            error_msg = (
                f"Invalid value: {savemapcoords}. "
                "Must be one of {', '.join(SaveMapCoords.__members__)}"
            )
            raise ValueError(error_msg) from err
    elif isinstance(savemapcoords, SaveMapCoords):
        return savemapcoords
    else:
        error_msg = (
            f"Invalid type for units: {type(savemapcoords)}. Must be SaveMapCoords enum or str."
        )
        raise TypeError(error_msg)
