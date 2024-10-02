"""Test the unit conversion."""

import numpy as np
import pytest
from pint import Quantity, UnitRegistry

from nanofinderparser.units import Units, convert_spectral_units

ureg = UnitRegistry()


@pytest.mark.parametrize(
    ("unit_in", "unit_out"),
    [
        ("nm", "cm-1"),
        ("cm-1", "eV"),
        ("eV", "raman_shift"),
        ("raman_shift", "nm"),
        ("nm", "nm"),
        ("cm-1", "cm-1"),
    ],
)
def test_convert_spectral_units(unit_in: Units, unit_out: Units) -> None:
    """Test the conversion of spectral units."""
    # TODO Test also for the case in which 'value' is a pint Quantity
    # Define the input values and the laser wavelength
    float_value = 500.0
    array_value = np.array([500.0, 600.0, 700.0])
    laser_nm = 532.000006769476 * ureg.nm  # Will use the own registry upon conversion

    # Test float input
    float_converted = convert_spectral_units(float_value, unit_in, unit_out, laser_nm)
    float_converted_back = convert_spectral_units(float_converted, unit_out, unit_in, laser_nm)
    assert isinstance(float_converted, float)
    assert isinstance(float_converted_back, float)
    assert np.isclose(float_converted_back, float_value, atol=1e-6)

    # Test array input
    array_converted = convert_spectral_units(array_value, unit_in, unit_out, laser_nm)
    array_converted_back = convert_spectral_units(array_converted, unit_out, unit_in, laser_nm)
    assert isinstance(array_converted, np.ndarray)
    assert isinstance(array_converted_back, np.ndarray)
    assert np.allclose(array_converted_back, array_value, atol=1e-6)

    if unit_in != "raman_shift":
        # Test Quantity input
        quantity_value = 500.0 / ureg.cm if unit_in == "cm-1" else 500.0 * ureg(unit_in)
        quantity_converted = convert_spectral_units(quantity_value, unit_in, unit_out, laser_nm)
        quantity_converted_back = convert_spectral_units(
            quantity_converted, unit_out, unit_in, laser_nm
        )
        assert isinstance(quantity_converted, Quantity)
        assert isinstance(quantity_converted_back, Quantity)
        assert np.allclose(quantity_converted_back.magnitude, quantity_value.magnitude, atol=1e-6)
    else:
        # Test proper raman_shift conversion
        zero_raman = convert_spectral_units(0, "raman_shift", "nm", laser_nm)
        assert zero_raman == laser_nm.magnitude

        # Raman works with laser not in nm units
        laser_m = 0.000000532000006769476 * ureg.m
        zero_raman = convert_spectral_units(0, "raman_shift", "nm", laser_m)
        assert np.allclose(zero_raman, laser_m.to("nm").magnitude)
