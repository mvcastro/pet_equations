import numpy as np
from numpy.typing import NDArray


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

    rnl = (
        sigma
        * 0.5
        * (np.power(tmax + 273.16, 4) + np.power(tmin + 273.16, 4))
        * (0.34 - 0.14 * np.sqrt(ea))
        * (1.35 * (rs / rso) - 0.35)
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

    ra = (24 * 60 / np.pi) * Gsc * rel_dist_es * (
            sha * np.sin(latitude) * np.sin(solar_dec)
            + np.cos(latitude) * np.cos(solar_dec) * np.sin(sha)
        )

    return ra