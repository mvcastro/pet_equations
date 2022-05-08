from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pet_equations.checking import _check_array_sizes, _check_param_type
from pet_equations.meteorological_vars import (daylight_hours, saturation_vapor_pressure,
                                 solar_declination, sunset_hour_angle)


def calculate(temp_c: NDArray[np.float64] | int | float,
              lat_deg: NDArray[np.float64] | int | float,
              doy: NDArray[np.int64] | int | float,
              d_hours: Optional[NDArray[np.float64] | int | float] = None
              ) -> NDArray[np.float64]:
    """Estimate of PET by Hamon equation - Hamon(1963).
       This function must receive 12 monthly values.

        PET = 0.1651 * Ld * RHOSAT * KPEC

        Where:
            Ld is the daytime length in multiples of 12 hours.
            KPEC is the calibration coefficient set to 1.

        # Saturated vapor density (g/m3) at the daily mean air temperature T (°C)
         RHOSAT = 216.7 x ESAT / (T + 273.3)

        # Saturated vapor pressure (mb) at the given T (°C)
        ESAT = 6.108 x EXP (17.26939 x T / (T + 237.3))

        Source: Lu et al. (2005). A comparison of six potential evaportranspiration
        methods for regional use in the southeastern United States.
        Journal of the American Water Resources Association, 41, 621-633.

    Args:
        temp_c (Union[np.ndarray, float]): daily mean air temperature (°C)
        lat_deg (Union[np.ndarray, float]): _description_
        date (np.ndarray, optional): _description_. Defaults to None.
        doy (Union[np.ndarray, float, None], optional): _description_. Defaults to None.
        d_hours (Union[np.ndarray, float, None], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        Union[np.ndarray, np.float64]: _description_
    """

    kpec = 1

    temp_c = _check_param_type(temp_c)
    lat_deg = _check_param_type(lat_deg)
    doy = _check_param_type(doy)
    d_hours = _check_param_type(d_hours)

    max_size = max(temp_c.size, doy.size, d_hours.size)

    _check_array_sizes([temp_c, doy, d_hours], max_size)

    svp = saturation_vapor_pressure(temp_c)

    if d_hours is None:
        lat_rad = np.deg2rad(lat_deg)
        solar_dec = solar_declination(doy)  # type: ignore
        sha = sunset_hour_angle(lat_rad, solar_dec)
        d_hours = daylight_hours(sha)

    pet = np.where(temp_c < 0.0, 0.0,
                   kpec * 0.165 * (d_hours / 12) * 216.7 * (svp / (temp_c + 273.3)))

    return pet
