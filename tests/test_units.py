import numpy as np
import pytest
from nanofinderparser.units import convert_spectral_units, setup_spectroscopy_constants


def test_setup_spectroscopy_constants(unit_registry):
    units_dict = setup_spectroscopy_constants(unit_registry)
    assert "nm" in units_dict
    assert "cm_1" in units_dict
    assert "raman_shift" in units_dict
    assert "eV" in units_dict


@pytest.mark.parametrize(
    "value, unit_in, unit_out, expected",
    [
        (500, "nm", "cm_1", 20000),
        (20000, "cm_1", "nm", 500),
        (2.48, "eV", "nm", 500),
        (500, "nm", "eV", 2.48),
        (500, "nm", "raman_shift", 18796.99),
        (3361, "raman_shift", "nm", 630),
    ],
)
def test_convert_spectral_units(value, unit_in, unit_out, expected):
    result = convert_spectral_units(value, unit_in, unit_out)
    assert np.isclose(result, expected, rtol=1e-2)


def test_convert_spectral_units_array():
    values = np.array([400, 500, 600])
    result = convert_spectral_units(values, "nm", "cm_1")
    expected = np.array([25000, 20000, 16666.67])
    assert np.allclose(result, expected, rtol=1e-2)


def test_convert_spectral_units_quantity(unit_registry):
    value = 500 * unit_registry.nm
    result = convert_spectral_units(value, "nm", "cm_1")
    assert np.isclose(result.magnitude, 20000, rtol=1e-2)
    assert str(result.units) == "1 / centimeter"


def test_convert_spectral_units_invalid_unit():
    with pytest.raises(ValueError):
        convert_spectral_units(500, "invalid_unit", "nm")


def test_convert_spectral_units_same_unit():
    value = 500
    result = convert_spectral_units(value, "nm", "nm")
    assert result == value
