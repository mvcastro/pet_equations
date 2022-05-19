"""Hargreaves equation - Emperical (Temperature-Based)

        Source: Allen et al. (1998). Crop evapotranspiration (guidelines for
         computing crop water requirements. FAO Irrigation and Drainage Paper 56.
         United Nations, Rome.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pet_equations import meteorological_vars as mv
from pet_equations.checking import (_check_array_sizes, _check_latitude,
                                    _check_param_type)


def calculate(latitude: NDArray[np.float64] | pd.Series | int | float,
              tmin: NDArray[np.float64] | pd.Series | int | float,
              tmax: NDArray[np.float64] | pd.Series | int | float,
              tmean: NDArray[np.float64] | pd.Series | int | float,
              doy: NDArray[np.int64] | pd.Series | int | float) -> NDArray[np.float64]:
    """Hargreaves equation - Emperical (Temperature-Based)

        ETo = 0.0023 * (Tmean + 17.8) * ((Tmax - Tmin) ** 0.5) *  Ra

        Source: Allen et al. (1998). Crop evapotranspiration (guidelines for
         computing crop water requirements. FAO Irrigation and Drainage Paper 56.
         United Nations, Rome.

    Args:
        latitude (NDArray[np.float64]): latitude (decimal degrees)
        tmin (NDArray[np.float64]): Minimum Temperatue (°C)
        tmax (NDArray[np.float64]): Maximum Temperature (°C)
        tmean (NDArray[np.float64]): Mean air temperature (°C)
        doy (NDArray[np.float64]): Day of Year betweem 1 and 365 or 366.

    Returns:
        NDArray[np.float64]: ETo (mm/day).
    """

    latitude = _check_param_type(latitude)
    tmin = _check_param_type(tmin)
    tmax = _check_param_type(tmax)
    tmean = _check_param_type(tmean)
    doy = _check_param_type(doy)

    max_size = max(latitude.size, tmin.size, tmax.size, doy.size)

    _check_latitude(latitude, max_size)
    _check_array_sizes([tmin, tmax, tmean, doy], max_size)

    # Latitude - Degrees to radians
    lat_rad = np.deg2rad(latitude)

    # Solar Declination (radians)
    solar_dec = mv.solar_declination(doy)

    # sunset hour angle (radians)
    sha = mv.sunset_hour_angle(lat_rad, solar_dec)

    # Relative distance earth-sun
    rel_dist_es = mv.relative_distance_earth_sun(doy)

    # Extra-Terrestrial Radiation (mm/day)
    et_ra = mv.extra_terrestrial_radiation(
        rel_dist_es, sha, lat_rad, solar_dec)

    # type: ignore
    return 0.0023 * et_ra * np.sqrt(tmax - tmin) * (tmean + 17.8)
