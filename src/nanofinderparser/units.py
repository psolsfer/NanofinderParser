# ISSUE #14 # TODO Handle units

# Add just 'nm', 'eV' and 'cm-1' and convert between them the spectral axis
# ALSO NEEDS TO CHANGE THE column names of data/processed data/baseline/fit...
# OTHER THAN THE SPECTRAL AXIS, needs to change the spectral range in the dataset...
import logging
from enum import Enum
from typing import Any, TypeVar, overload

import numpy as np
from numpy.typing import NDArray
from pint import Quantity, Unit, UnitRegistry

logger = logging.getLogger(__name__)
FloatOrArray = TypeVar("FloatOrArray", float, NDArray[np.float64])


class Units(str, Enum):
    nm = "nm"
    cm_1 = "cm_1"
    ev = "eV"
    raman_shift = "raman_shift"


def validate_units(units: Units | str | Any) -> Units:
    """Convert string to Units enum if necessary and validate the input.

    Parameters
    ----------
    units : Units or str
        The units to check and potentially convert.

    Returns
    -------
    Units
        The validated Units enum value.

    Raises
    ------
    ValueError
        If the input is not a valid Units enum value or string representation.
    """
    if isinstance(units, str):
        try:
            return Units(units.lower())
        except ValueError as err:
            raise ValueError(
                f"Invalid units value: {units}. " f"Must be one of {', '.join(Units.__members__)}"
            ) from err
    elif isinstance(units, Units):
        return units
    else:
        raise TypeError(f"Invalid type for units: {type(units)}. " f"Must be Units enum or str.")


def setup_spectroscopy_constants(
    registry: UnitRegistry,
) -> dict[str, Unit]:
    """Set up constants and units for spectroscopy calculations.

    Parameters
    ----------
    registry : UnitRegistry
        The Pint UnitRegistry to use for creating quantities.

    Returns
    -------
    dict[str, Unit]
        Dictionary of spectroscopy units
    """
    # Enable conversions relevant to spectroscopy
    registry.enable_contexts("spectroscopy")

    units_dict: dict[str, Unit] = {
        "nm": registry.nm,
        "cm_1": 1 / registry.cm,
        "raman_shift": 1 / registry.cm,
        "eV": registry.eV,
    }

    # h = registry.planck_constant
    # c = registry.speed_of_light
    # hc = h * c

    return units_dict


@overload
def convert_spectral_units(
    value: FloatOrArray,
    unit_in: Units | str,
    unit_out: Units | str,
    laser_wavelength_nm: float | Quantity = 532.000006769476,
) -> FloatOrArray: ...


@overload
def convert_spectral_units(
    value: Quantity,
    unit_in: Units
    | str,  # FIXME Could be inferred from 'value' (value.units) but not for raman_shift...
    unit_out: Units | str,
    laser_wavelength_nm: float | Quantity = 532.000006769476,
) -> Quantity: ...


def convert_spectral_units(
    value: FloatOrArray | Quantity,
    unit_in: Units | str,
    unit_out: Units | str,
    laser_wavelength_nm: float | Quantity = 532.000006769476,
) -> FloatOrArray | Quantity:
    """Convert spectral data between different units.

    Parameters
    ----------
    value : float | np.ndarray | Quantity
        The spectral data to convert.
    unit_in : {"nm", "cm_1", "eV", "raman_shift"}
        The unit of the input data.
    unit_out : {"nm", "cm_1", "eV", "raman_shift"}
        The unit to convert the data to.
    laser_wavelength_nm : float | Quantity
        The wavelength of the laser, used to properly convert to the Raman shift in cm-1.
        When passed as a float, the units must be in 'nm'. For a Quantity, it doesn't matter the
        specific unit used.

    Returns
    -------
    float or np.ndarray or pint.Quantity
        The converted spectral data. The return type matches the input type of `value`:
        - If `value` is a float or np.ndarray, returns a float or np.ndarray.
        - If `value` is a pint.Quantity, returns a pint.Quantity.


    Raises
    ------
    ValueError
        If `unit_in` or `unit_out` is not one of {"nm", "cm_1", "eV", "raman_shift"}.
    """
    # TODO Raman shift can't be passed as a Quantity

    if unit_in == unit_out:
        return value

    unit_in = validate_units(unit_in)
    unit_out = validate_units(unit_out)

    # Uses the registry from any given Quantity
    if isinstance(value, Quantity):
        registry = value._REGISTRY
    elif isinstance(laser_wavelength_nm, Quantity):
        registry = laser_wavelength_nm._REGISTRY
    else:
        registry = UnitRegistry()

    units_dict = setup_spectroscopy_constants(registry)

    if not isinstance(laser_wavelength_nm, Quantity):
        laser_wavelength_nm = laser_wavelength_nm * registry.nm

    input_is_quantity = isinstance(value, Quantity)
    if not input_is_quantity:
        try:
            value = value * units_dict[unit_in]
        except KeyError:
            msg = f"Invalid input unit: {unit_in}"
            raise ValueError(msg) from KeyError

    # TODO Try to implement this conversion in a pint's context in which the laser wavelength is
    # passed https://pint.readthedocs.io/en/0.23/user/contexts.html#working-without-a-default-definition
    if unit_in == "raman_shift":
        # Raman shift to cm-1
        value = laser_wavelength_nm.to(registry.cm**-1) - value

    if unit_out == "raman_shift":
        converted_value = laser_wavelength_nm.to(registry.cm**-1) - value.to(registry.cm**-1)
    else:
        converted_value = value.to(unit_out)

    if not input_is_quantity:
        converted_value = converted_value.magnitude

    return converted_value
