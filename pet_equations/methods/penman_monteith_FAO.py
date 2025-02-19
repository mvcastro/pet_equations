import numpy as np
from numpy.typing import NDArray

import pet_equations.methods_parameters.astronomical_variables as avars
import pet_equations.methods_parameters.meteorological_variables as mvars
import pet_equations.methods_parameters.radiation_variables as rvars


def calculate(
    tmax: NDArray[np.floating],
    tmin: NDArray[np.floating],
    tmean: NDArray[np.floating] | None,
    rn: NDArray[np.floating],
    rhmean: NDArray[np.floating],
    u2: NDArray[np.floating],
    altitude: float,
    latitude: float,
    doy: NDArray[np.int_],
    g: NDArray[np.floating] | float = 0,
) -> NDArray[np.floating]:
    if not tmean:
        tmean = np.array((tmax + tmin) / 2)

    # rns = net_shortwave_radiation(rs=rs)
    # rel_dist_es = avars.relative_distance_earth_sun(doy=doy)
    # solar_dec = avars.solar_declination(doy=doy)
    # sha = avars.sunset_hour_angle(latitude, solar_dec)
    # ra = rvars.extra_terrestrial_radiation(rel_dist_es, sha, latitude, solar_dec)
    # rso = clear_sky_shortwave_radiation(altitude, ra)
    es = mvars.saturation_vapor_pressure(temperature=tmean)
    ea = mvars.actual_vapour_pressure(es, rhmean)
    # rnl = net_longwave_radiation(tmax, tmin, rs, rso, ea)
    # rn = rns - rnl
    atm_p = mvars.atmospheric_pressure(altitude)
    psi_const = mvars.psychrometric_constant(atm_p)
    delta = mvars.slope_saturation_vapour_pressure_curve(temperature=tmean)

    eto = (
        0.408 * delta * (rn - g) + psi_const * 900.0 * u2 * (es - ea) / (tmean + 273)
    ) / (delta + psi_const * (1 + 0.34 * u2))

    return eto
