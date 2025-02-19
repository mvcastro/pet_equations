import numpy as np
from numpy.typing import NDArray


def daylight_hours(sha: NDArray[np.float64]) -> NDArray[np.float64]:
    """Estimate of daylight hours N.
       Source: Allen et al. (1998). Crop evapotranspiration (guidelines for
        computing crop water requirements. FAO Irrigation and Drainage Paper 56.
        United Nations, Rome.

    Args:
        sha (NDArray[np.float64]): sunset hour angle [radians].

    Returns:
        NDArray[np.float64]: daylight hours [hours].
    """

    return (24 / np.pi) * sha


def sunset_hour_angle(
    latitude: NDArray[np.float64], solar_dec: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sunset Hour Angle ωs
       Source: Allen et al. (1998). Crop evapotranspiration (guidelines for
        computing crop water requirements. FAO Irrigation and Drainage Paper 56.
        United Nations, Rome.

    Args:
        latitude (NDArray[np.float64]): latitude of site [radians].
        solar_dec (NDArray[np.float64]): solar declination d [radians].

    Returns:
        NDArray[np.float64]: Sunset Hour Angle (radians)
    """
    sha = np.arccos(-np.tan(latitude) * np.tan(solar_dec))
    return np.array(sha)


def relative_distance_earth_sun(doy: NDArray[np.int64]) -> NDArray[np.float64]:
    """Inverse Relative Distance Earth-Sun - dr (radians).
        Source: Allen et al. (1998). Crop evapotranspiration (guidelines for
         computing crop water requirements. FAO Irrigation and Drainage Paper 56.
         United Nations, Rome.

    Args:
        doy (NDArray[np.int64]): day of year between 1 and 365 or 366.

    Returns:
        NDArray[np.float64]: dr [radians]
    """
    return 1 + 0.033 * np.cos(2 * np.pi * doy / 365)


def solar_declination(doy: NDArray[np.int64]) -> NDArray[np.float64]:
    """Solar Declination δ (radians).
        Source: Allen et al. (1998). Crop evapotranspiration (guidelines for
         computing crop water requirements. FAO Irrigation and Drainage Paper 56.
         United Nations, Rome.

    Args:
        doy (NDArray[np.int64]): day of year between 1 and 365 or 366.

    Returns:
         NDArray[np.float64]: solar declination (radians).
    """
    solar_dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    return solar_dec





def is_leap_year(year: int) -> bool:
    """Determine whether a year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
