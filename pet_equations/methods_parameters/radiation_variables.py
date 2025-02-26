from typing import Literal

import numpy as np
from numpy.typing import NDArray


def solar_radiation(
    n: NDArray[np.floating],
    N: NDArray[np.floating],
    ra: NDArray[np.floating],
    a_s: float = 0.25,
    b_s: float = 0.5,
) -> NDArray[np.floating]:
    """Solar or Shortwave Radiation (Rs)

    Args:
        n (NDArray[np.floating]): actual duration of sunshine [hour]
        N (NDArray[np.floating]): maximum possible duration of sunshine or daylight hours [hour]
        ra (NDArray[np.floating]): extraterrestrial radiation [MJ m-2 day-1]
        a_s (float, optional): regression constant, expressing the fraction of extraterrestrial radiation 
            reaching the earth on overcast days (n = 0). Defaults to 0.25.
        b_s (float, optional): _description_. Defaults to 0.5.

    Returns:
        NDArray[np.floating]: solar or shortwave radiation [MJ m-2 day-1]
    """
    rs = (a_s + b_s * (n / N)) * ra
    return rs


def net_shortwave_radiation(
    rs: NDArray[np.floating], alpha: float = 0.23
) -> NDArray[np.floating]:
    """Net Solar or Shortwave Radiation (Rns)

    Args:
        rs (NDArray[np.floating]): Solar or Shortwave Radiation [MJ/m2.day]
        alpha (float, optional): albedo [-]. Defaults to 0.23.

    Returns:
        NDArray[np.floating]: Net Solar or Short Wave Radiation [MJ/m2.day]
    """
    rns = (1 - alpha) * rs
    return rns


def clear_sky_shortwave_radiation(
    altitude: float, ra: NDArray[np.floating]
) -> NDArray[np.floating]:
    """clear-sky solar or clear-sky shortwave radiation (Rso)

    Args:
        altitude (float): Altitude - z [m]
        ra (NDArray[np.floating]): Extraterrestrial Radiation - Ra [MJ/m.day]

    Returns:
        NDArray[np.floating]: clear-sky solar or clear-sky shortwave radiation [MJ/m2.day]
    """
    rso = (0.75 + 2e-5 * altitude) * ra
    return rso


def net_longwave_radiation(
    tmax: NDArray[np.floating],
    tmin: NDArray[np.floating],
    rs: NDArray[np.floating],
    rso: NDArray[np.floating],
    ea: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Net Longwave Radiation (Rnl)

    Args:
        tmax (NDArray[np.floating]): daily maximum air temperature [°C]
        tmin (NDArray[np.floating]): daily minimum air temperature [°C]
        rs (NDArray[np.floating]): solar or shortwave radiation [MJ/m2.day]
        rso (NDArray[np.floating]): clear-sky solar or clear-sky shortwave radiation [MJ/m2.day]
        ea (NDArray[np.floating]): actual vapour pressure [kPa]

    Returns:
        NDArray[np.floating]: net longwave radiation [MJ/m2.day]
    """

    # Stefan-Boltzmann constant (σ) [MJ/K4.m2.day]
    sigma = 4.903e-9
    
    ratio_rs_rso = np.where(rs / rso <= 1.0,  rs / rso, 1.0)

    rnl = (
        sigma
        * 0.5
        * (np.power(tmax + 273.16, 4) + np.power(tmin + 273.16, 4))
        * (0.34 - 0.14 * np.sqrt(ea))
        * (1.35 * (ratio_rs_rso) - 0.35)
    )
    return rnl


def extra_terrestrial_radiation(
    rel_dist_es: NDArray[np.float64],
    sha: NDArray[np.float64],
    latitude: NDArray[np.float64] | float,
    solar_dec: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Extra-Terrestrial Radiation (Ra)
       Source: Allen et al. (1998). Crop evapotranspiration (guidelines for
       computing crop water requirements. FAO Irrigation and Drainage Paper 56.
       United Nations, Rome.

    Args:
        rel_dist_es (NDArray[np.float64]): relative distance earth-sun.
        sha (NDArray[np.float64]): sunset hour angle [radians]
        latitude (NDArray[np.float64]): latitude of site [radians]
        solar_dec (NDArray[np.float64]): solar declination [radians]

    Returns:
        NDArray[np.float64]: Extra-Terrestrial Radiation (MJ.m-2.day-1).
    """
    
    # solar constant [MJ.m-2.min-1]
    Gsc = 0.0820

    ra = (24.0 * 60.0 / np.pi) * Gsc * rel_dist_es * (
            sha * np.sin(latitude) * np.sin(solar_dec)
            + np.cos(latitude) * np.cos(solar_dec) * np.sin(sha)
        )

    return ra


def soil_heat_flux(
    temp_iminus1: NDArray[np.float64],
    temp_i: NDArray[np.float64],
    temp_iplus1: NDArray[np.float64] | None,
    time_interval: Literal['daily', 'dekadal', 'monthly']
) -> NDArray[np.float64]:
    """Soil Heat Flux

    Args:
        temp_iminus1 (NDArray[np.float64]): air temperature at time i-1 [°C]
        temp_i (NDArray[np.float64]): air temperature at time i [°C]
        temp_iplus1 (NDArray[np.float64] | None): air temperature at time i+1 [°C]
        time_interval (Literal[&#39;daily&#39;, &#39;dekadal&#39;, &#39;monthly&#39;, &#39;hourly&#39;]): _description_

    Returns:
        NDArray[np.float64]: soil heat flux [MJ m-2 day-1]
    """
    
    # G ~= 0
    if time_interval in ['daily', 'dekadal']:
        return np.array(0.0)
    
    if time_interval == 'monthly':
        if temp_iplus1 is None:
            g = 0.14 * (temp_i - temp_iminus1)
            return g
            
        g = 0.07 * (temp_iplus1 - temp_iminus1)
        return g

    
def soil_heat_flux_for_hourly_or_short_periods(
    rn: NDArray[np.float64],
    period: Literal['daylight', 'nighttime']
):
    if period == 'daylight':
        g = 0.1 * rn
        return g
    
    if period == 'nighttime':
        g = 0.5 * rn
        return g
        

    
    