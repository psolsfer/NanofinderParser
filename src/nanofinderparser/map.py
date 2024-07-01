"""Functions related with Nanofinder mappings."""

import numpy as np
import pandas as pd


def _nanofinder_mapcoords(x_size: int, y_size: int) -> pd.DataFrame:
    """Generate map coordinates from the size of the x and y dimensions.

    This function creates a DataFrame of (x, y) coordinates corresponding to the order in which
    spectra are obtained from raw NanoFinder smd files.

    Parameters
    ----------
    x_size : int
        The number of points along the x-axis.
    y_size : int
        The number of points along the y-axis.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns:
        - 'x': x-coordinates (int)
        - 'y': y-coordinates (int)
        The DataFrame has x_size * y_size rows, ordered as follows:
        (0,0), (1,0), ..., (x_size-1,0), (0,1), (1,1), ..., (x_size-1,y_size-1)

    Notes
    -----
    The coordinates are generated in the following order:
    x           | y
    ---------------
    0           | 0
    1           | 0
    2           | 0
    ...
    (x_size-1)  | 0
    0           | 1
    ...
    (x_size-1)  | (y_size-1)

    Examples
    --------
    >>> _nanofinder_mapcoords(3, 2)
       x  y
    0  0  0
    1  1  0
    2  2  0
    3  0  1
    4  1  1
    5  2  1
    """
    x = np.tile(np.arange(x_size), y_size)
    y = np.repeat(np.arange(y_size), x_size)

    return pd.DataFrame({"x": x, "y": y})
