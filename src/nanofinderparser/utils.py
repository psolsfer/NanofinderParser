"""Utilities."""

import operator
from functools import reduce
from typing import Any


def get_nested_dict_value(data: dict[str, Any], keys: str) -> Any:
    """Safely get a value from a nested dictionary.

    Parameters
    ----------
    data : Dict[str, Any]
        The nested dictionary.
    keys : str
        Dot-separated string of keys to access the nested value.

    Returns
    -------
    Any
        The value at the specified nested location.

    Raises
    ------
    KeyError
        If any key in the path doesn't exist.
    """
    return reduce(operator.getitem, keys.split("."), data)
