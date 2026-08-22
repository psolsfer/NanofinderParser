"""Utilities."""

from enum import StrEnum
from typing import Any, Final

# NanoFinder is written in Visual Basic, whose booleans are -1 for true and 0 for false.
VB_TRUE: Final[str] = "-1"
VB_FALSE: Final[str] = "0"


def parse_vb_bool(value: str | bool | int) -> bool:
    """Parse the Visual Basic boolean convention used by NanoFinder files.

    Parameters
    ----------
    value : str | bool | int
        Raw value from the file: ``"0"``/``0`` for false, ``"-1"``/``-1`` for true. Booleans and
        other integers are converted as they are.

    Returns
    -------
    bool
        The parsed value.

    Raises
    ------
    ValueError
        If a string value other than ``"0"`` or ``"-1"`` is encountered.
    """
    if isinstance(value, str):
        if value == VB_FALSE:
            return False
        if value == VB_TRUE:
            return True
        msg = f"Unexpected boolean string value: {value!r}; expected {VB_FALSE!r} or {VB_TRUE!r}"
        raise ValueError(msg)
    return bool(value)


def format_vb_bool(value: bool) -> str:
    """Write a boolean the way NanoFinder files store it.

    Parameters
    ----------
    value : bool
        The value to write.

    Returns
    -------
    str
        ``"-1"`` for true and ``"0"`` for false.
    """
    return VB_TRUE if value else VB_FALSE


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
