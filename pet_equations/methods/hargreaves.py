"""Hargreaves equation - Emperical (Temperature-Based)

Source: Allen et al. (1998). Crop evapotranspiration (guidelines for
 computing crop water requirements. FAO Irrigation and Drainage Paper 56.
 United Nations, Rome.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

import pet_equations.methods_parameters.astronomical_variables as avars
import pet_equations.methods_parameters.radiation_variables as rvars
from pet_equations.checking import (
    _check_array_sizes,
    _check_latitude,
    _check_param_float_type,
    _check_param_int_type,
)


def calculate(
    latitude: ArrayLike | int | float,
    tmin: ArrayLike | int | float,
    tmax: ArrayLike | int | float,
    tmean: ArrayLike | int | float,
    doy: ArrayLike | int,
) -> NDArray[np.float64]:
    """Hargreaves equation - Emperical (Temperature-Based)

        ETo = 0.0023 * (Tmean + 17.8) * ((Tmax - Tmin) ** 0.5) * Ra

        Source: Allen et al. (1998). Crop evapotranspiration (guidelines for
         computing crop water requirements. FAO Irrigation and Drainage Paper 56.
         United Nations, Rome.

    Args:
        latitude (NDArray[np.float64]): latitude (decimal degrees)
        tmin (NDArray[np.float64]): Minimum Temperatue (°C)
        tmax (NDArray[np.float64]): Maximum Temperature (°C)
        tmean (NDArray[np.float64]): Mean air temperature (°C)
        doy (NDArray[np.int64]): Day of Year betweem 1 and 365 or 366.

    Returns:
        NDArray[np.float64]: ETo (mm/day).
    """

    latitude = _check_param_float_type(latitude)
    tmin = _check_param_float_type(tmin)
    tmax = _check_param_float_type(tmax)
    tmean = _check_param_float_type(tmean)
    doy = _check_param_int_type(doy)

    max_size = max(latitude.size, tmin.size, tmax.size, doy.size)

    _check_latitude(latitude, max_size)
    _check_array_sizes([tmin, tmax, tmean, doy], max_size)

    # Latitude - Degrees to radians
    lat_rad = np.deg2rad(latitude)

    # Solar Declination (radians)
    solar_dec = avars.solar_declination(doy)

    # sunset hour angle (radians)
    sha = avars.sunset_hour_angle(lat_rad, solar_dec)

    # Relative distance earth-sun
    rel_dist_es = avars.inverse_relative_distance_earth_sun(doy)

    # Extra-Terrestrial Radiation (mm/day)
    et_ra = rvars.extra_terrestrial_radiation(
        rel_dist_es=rel_dist_es, sha=sha, latitude=lat_rad, solar_dec=solar_dec
    )

    return 0.0023 * (et_ra * 0.408) * np.sqrt(tmax - tmin) * (tmean + 17.8)
