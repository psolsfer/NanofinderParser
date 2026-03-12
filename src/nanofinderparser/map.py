"""Functions related with Nanofinder mappings."""

import numpy as np
import pandas as pd


def _nanofinder_mapcoords(
    x_size: int,
    y_size: int,
    x_start: float = 0.0,
    y_start: float = 0.0,
    x_step: float = 1.0,
    y_step: float = 1.0,
) -> pd.DataFrame:
    """Generate map coordinates from the size of the x and y dimensions.

    This function creates a DataFrame of (x, y) coordinates corresponding to the order in which
    spectra are obtained from raw NanoFinder smd files.  When physical start/step values are
    provided the coordinates are in the same physical units as the stage axes (typically nm).

    Parameters
    ----------
    x_size : int
        The number of points along the x-axis.
    y_size : int
        The number of points along the y-axis.
    x_start : float, optional
        Physical start coordinate of the x-axis, by default 0.0.
    y_start : float, optional
        Physical start coordinate of the y-axis, by default 0.0.
    x_step : float, optional
        Physical step size along the x-axis, by default 1.0.
    y_step : float, optional
        Physical step size along the y-axis, by default 1.0.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns:
        - 'x': x-coordinates (float)
        - 'y': y-coordinates (float)
        The DataFrame has x_size * y_size rows, ordered as follows:
        (x_start, y_start), (x_start+x_step, y_start), ..., advancing x first (fast axis).

    Notes
    -----
    The coordinates are generated in x-fast (row-major) order:

    x                          | y
    ---------------------------------------
    x_start                    | y_start
    x_start + x_step           | y_start
    x_start + 2*x_step         | y_start
    ...
    x_start + (x_size-1)*x_step| y_start
    x_start                    | y_start + y_step
    ...

    Examples
    --------
    >>> _nanofinder_mapcoords(3, 2)
         x    y
    0  0.0  0.0
    1  1.0  0.0
    2  2.0  0.0
    3  0.0  1.0
    4  1.0  1.0
    5  2.0  1.0
    """
    xi = np.tile(np.arange(x_size), y_size)
    yi = np.repeat(np.arange(y_size), x_size)
    return pd.DataFrame({"x": x_start + xi * x_step, "y": y_start + yi * y_step})
