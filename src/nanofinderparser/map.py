"""Functions related with Nanofinder mappings."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pint_pandas  # noqa: F401


@dataclass(frozen=True, slots=True)
class AxisSpec:
    """Physical scan specification for a single stage axis.

    Groups the start position, step size, and units needed to generate physical map
    coordinates along one axis.

    Attributes
    ----------
    start : float
        Physical start coordinate of the axis, by default 0.0.
    step : float
        Physical step size along the axis, by default 1.0.
    units : str | None
        Units for the axis, by default None.
    """

    start: float = 0.0
    step: float = 1.0
    units: str | None = None


def _nanofinder_mapcoords(
    x_size: int,
    y_size: int,
    *,
    x_axis: AxisSpec | None = None,
    y_axis: AxisSpec | None = None,
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
    x_axis : AxisSpec | None, optional
        Physical start, step, and units for the x-axis, by default None (start=0.0, step=1.0,
        units=None).
    y_axis : AxisSpec | None, optional
        Physical start, step, and units for the y-axis, by default None (start=0.0, step=1.0,
        units=None).

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
    x_axis = x_axis or AxisSpec()
    y_axis = y_axis or AxisSpec()

    xi = np.tile(np.arange(x_size), y_size)
    yi = np.repeat(np.arange(y_size), x_size)
    x_dtype = f"pint[{x_axis.units}]" if x_axis.units else None
    y_dtype = f"pint[{y_axis.units}]" if y_axis.units else None
    return pd.DataFrame(
        {
            "x": pd.Series(x_axis.start + xi * x_axis.step, dtype=x_dtype),
            "y": pd.Series(y_axis.start + yi * y_axis.step, dtype=y_dtype),
        }
    )
