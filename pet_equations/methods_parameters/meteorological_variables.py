import numpy as np
from numpy.typing import NDArray

# =======================
# ATMOSPHERIC PARAMETERS
# =======================

def atmospheric_pressure(
    altitude: float | NDArray[np.floating],
) -> NDArray[np.floating]:
    """Atmospheric Pressure dependent of Altitude (z)
       Source: Allen et al. (1998). Crop evapotranspiration (guidelines for
       computing crop water requirements. FAO Irrigation and Drainage Paper 56.
       United Nations, Rome.

    Args:
        altitude (NDArray): Altitude [m]

    Returns:
        NDArray[np.]: Atmospheric pressure [KPa]
    """
    atm_p = 101.3 * np.power((293.0 - 0.0065 * altitude) / 293.0, 5.26)
    return atm_p


def psychrometric_constant(
    atmospheric_pressure: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Psychrometric Constant (γ)
       Source: Allen et al. (1998). Crop evapotranspiration (guidelines for
       computing crop water requirements. FAO Irrigation and Drainage Paper 56.
       United Nations, Rome.

    Args:
        atmospheric_pressure (NDArray[np.floating]): Atmospheric pressure (KPa)

    Returns:
        NDArray[np.float]: Psychrometric Constant  [kPa/°C]
    """
    psi_const = 0.665e-3 * atmospheric_pressure
    return psi_const


def saturation_vapor_pressure(
    temperature: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Saturation Vapour Pressure (es).
        Source: Allen et al. (1998). Crop evapotranspiration (guidelines for
        computing crop water requirements. FAO Irrigation and Drainage Paper 56.
        United Nations, Rome.

    Args:
        temperature (NDArray[np.float64]): air temperature [°C]

    Returns:
        NDArray[np.floating]: saturation vapour pressure for a given time period [kPa]
    """
    es = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))
    return es


def mean_saturation_vapor_pressure(
    tmin: NDArray[np.floating], tmax: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Mean Saturation Vapour Pressure (es).
        Source: Allen et al. (1998). Crop evapotranspiration (guidelines for
        computing crop water requirements. FAO Irrigation and Drainage Paper 56.
        United Nations, Rome.

    Args:
        tmin (NDArray[np.floating]): Minimum air temperature [°C]
        tmax (NDArray[np.floating]): Maximum air temperature [°C]

    Returns:
        NDArray[np.floating]: saturation vapour pressure for a given time period [kPa]
    """
    mean_es = (saturation_vapor_pressure(tmin) + saturation_vapor_pressure(tmax)) / 2
    return mean_es


def slope_saturation_vapour_pressure_curve(
    temperature: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Slope Saturation Vapour Pressure Curve (∆)

    Args:
        temperature (NDArray): daily air temperature [°C]

    Returns:
        NDArray[np.floating]: Slope Saturation Vapour Pressure Curve  [kPa/°C]
    """
    delta = (
        4098
        * 0.6108
        * np.exp(17.27 * temperature / (temperature + 237.3))
        / np.power(temperature + 237.3, 2)
    )
    return delta


def actual_vapour_pressure(
    es: NDArray[np.floating], rhmean: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Actual Vapour Pressure (ea)
       Source: Allen et al. (1998). Crop evapotranspiration (guidelines for
       computing crop water requirements. FAO Irrigation and Drainage Paper 56.
       United Nations, Rome.

    Args:
        es (NDArray[np.floating]): mean saturation vapour pressure - es [kPa]
        rhmean (NDArray[np.floating]): daily mean relative humidity - RHmean [%]

    Returns:
        NDArray[np.floating]: actual vapour pressure [kPa]
    """
    ea = es * rhmean / 100
    return ea


def wind_speed_at_2m(
    uz: NDArray[np.floating],
    z: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Wind Speed at 2 m above ground surface [m.s-1]

    Args:
        uz (NDArray[np.floating]): measured wind speed at z m above ground surface [m s-1]
        z (NDArray[np.floating]): height of measurement above ground surface [m]

    Returns:
        NDArray[np.floating]: wind speed at 2 m above ground surface [m.s-1]
    """
    u2 = uz * 4.87 / np.log(67.8 * z - 5.42)
    return u2
    