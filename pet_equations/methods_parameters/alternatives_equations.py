import numpy as np
from numpy.typing import NDArray


def extra_terrestrial_radiation(
    rel_dist_es: NDArray[np.float64],
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

    et_ra = (15.392 * rel_dist_es * (
        sha * np.sin(latitude) * np.sin(solar_dec)
        + np.cos(latitude) * np.cos(solar_dec) * np.sin(sha)
    ))

    return et_ra


def sunset_hour_angle(
    latitude: NDArray[np.float64] | float, solar_dec: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sunset Hour Angle (ωs)

    Args:
        latitude (NDArray[np.float64]): latitude of site [radians].
        solar_dec (NDArray[np.float64]): solar declination d [radians].

    Returns:
        NDArray[np.float64]: Sunset Hour Angle (radians)
    """
    x = 1 - np.power(np.tan(latitude), 2) * np.power(np.tan(solar_dec), 2)
    x = x if x > 0.0 else 0.00001
    sha = (np.pi / 2) - np.arctan(-np.tan(latitude) * np.tan(solar_dec) / np.sqrt(x))
    return np.array(sha)