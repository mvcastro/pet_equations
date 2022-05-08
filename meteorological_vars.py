import numpy as np
from numpy.typing import NDArray


def saturation_vapor_pressure(temp: NDArray[np.float64]) -> NDArray[np.float64]:
    """Estimate of Saturation Vapor Pressure - es (mb).
       Source: Lu et al. (2005). A comparison of six potential evaportranspiration
        methods for regional use in the southeastern United States.
        Journal of the American Water Resources Association, 41, 621-633.

    Args:
        temp (NDArray[np.float64]): daily mean air temperature [°C].

    Returns:
        NDArray[np.float64]: Saturation Vapor Pressure [mb].
    """
    return 6.108 * np.exp(17.26939 * temp / (temp + 237.3))


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


def sunset_hour_angle(latitude: NDArray[np.float64],
                      solar_dec: NDArray[np.float64]
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


def extra_terrestrial_radiation(rel_dist_es: NDArray[np.float64],
                                sha: NDArray[np.float64],
                                latitude: NDArray[np.float64],
                                solar_dec: NDArray[np.float64],
                                ) -> NDArray[np.float64]:
    """Ra -> Extra-Terrestrial Radiation (mm/day).
        Source: Yates, D. & Strzepek (1994). Potential Evapotranspiration Methods
         and their Impact on the Assessment of River Basin Runoff Under Climate Change

    Args:
        rel_dist_es (NDArray[np.float64]): relative distance earth-sun.
        sha (NDArray[np.float64]): sunset hour angle [radians]
        latitude (NDArray[np.float64]): latitude of site [radians]
        solar_dec (NDArray[np.float64]): solar declination [radians]

    Returns:
        NDArray[np.float64]: Extra-Terrestrial Radiation (mm/day).
    """

    et_ra = 15.392 * rel_dist_es * (sha * np.sin(latitude) * np.sin(solar_dec) +
                                    np.cos(latitude) * np.cos(solar_dec) * np.sin(sha))

    return et_ra


def is_leap_year(year: int) -> bool:
    """Determine whether a year is a leap year."""

    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
