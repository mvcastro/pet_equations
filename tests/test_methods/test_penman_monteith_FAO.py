import unittest
from datetime import datetime

import numpy as np

import pet_equations.methods.penman_monteith_FAO as pmFAO
import pet_equations.methods_parameters.astronomical_variables as avars
import pet_equations.methods_parameters.radiation_variables as rvars
import pet_equations.methods_parameters.meteorological_variables as mvars


class TestPenmanMontheith(unittest.TestCase):
    """
    Test realized from the document: Allen et al. (1998). Crop evapotranspiration
    (guidelines for computing crop water requirements. FAO Irrigation and Drainage Paper 56.
    United Nations, Rome.
    """

    def test_eto_with_mean_monthly_data(self):
        """
        Determination of ETo with mean monthly data - Example 17 - pg.70
        Monthly average climatic data of April of Bangkok (Thailand)
        located at 13°44'N and at an elevation of 2 m
        """
        altitude = 2.0
        date = datetime(2023, 4, 15)
        doy = date.timetuple().tm_yday
        latitude_graus = 13.73
        
        # Monthly average daily maximum temperature [°C]
        tmax = 34.8
        # Monthly average daily minimum temperature [°C]
        tmin = 25.6
        # Monthly average daily vapour pressure [kPa]
        ea = 2.85
        # Monthly average daily wind speed [m/s]
        u2 = 2.0
        # Monthly average sunshine duration (n) [hours/day]
        actual_duration_of_sunshine = 8.5
        # Mean monthly average temperature (Tmonth,i-1)
        temp_march = 29.2
        # Mean monthly average temperature (Tmonth,i) [°C]
        temp_april = 30.2

        tmean = (tmax + tmin) / 2
        self.assertAlmostEqual(tmean, 30.2, delta=0.01)
        
        # Slope Saturation Vapour Pressure Curve
        delta = mvars.slope_saturation_vapour_pressure_curve(temperature=tmean)
        self.assertAlmostEqual(delta, 0.246, delta=0.1)
        
        # Atmospheric Pressure 
        atm_p = mvars.atmospheric_pressure(altitude)
        self.assertAlmostEqual(atm_p, 101.3, delta=0.1)
        
        # Psychrometric Constant
        psi_const = mvars.psychrometric_constant(atm_p)
        self.assertAlmostEqual(psi_const, 0.0674, delta=0.01)
        
        # Mean Saturation Vapour Pressure (es).
        es = mvars.mean_saturation_vapor_pressure(tmax=tmax, tmin=tmin)
        self.assertAlmostEqual(es, 4.42, delta=0.1)

        # Solar Radiation (Rs)
        latitude = np.deg2rad(latitude_graus)
        rel_dist_es = avars.inverse_relative_distance_earth_sun(doy)
        solar_dec = avars.solar_declination(doy)
        sha = avars.sunset_hour_angle(latitude, solar_dec)
        
        # Daylength (N)
        daylight_hours = avars.daylight_hours(sha=sha)
        self.assertAlmostEqual(daylight_hours, 12.31, delta=0.1)
        
        # Extraterrestriial Radiation (Ra)
        ra = rvars.extra_terrestrial_radiation(rel_dist_es, sha, latitude, solar_dec)
        self.assertAlmostEqual(ra, 38.06, delta=0.1)
        
        # Solar Radiation (Rs)
        rs = rvars.solar_radiation(
            n=actual_duration_of_sunshine, N=daylight_hours, ra=ra
        )
        self.assertAlmostEqual(rs, 22.65, delta=0.1)
        
        # Clear-Sky Solar or Clear-Sky Shortwave Radiation 
        rso = rvars.clear_sky_shortwave_radiation(
            altitude=altitude, ra=ra
        )
        self.assertAlmostEqual(rso, 28.54, delta=0.1)
        
        # Net Solar or Shortwave Radiation (Rns)
        rns = rvars.net_shortwave_radiation(rs=rs)
        self.assertAlmostEqual(rns, 17.44, delta=0.1)
        
        # Net Longwave Radiation (Rnl)
        rnl = rvars.net_longwave_radiation(
            tmax=tmax,
            tmin=tmin,
            rs=rs,
            rso=rso,
            ea=ea
        )
        self.assertAlmostEqual(rnl, 3.11, delta=0.1)
        
        # Net Radiation (Rn)
        rn = rns - rnl
        self.assertAlmostEqual(rn, 14.33, delta=0.1)
        
        # Soil Heat Flux
        g = rvars.soil_heat_flux(
            temp_iminus1=temp_march,
            temp_i=temp_april,
            temp_iplus1=None,
            time_interval='monthly'
        )
        self.assertAlmostEqual(g, 0.14, delta=0.1)      
        
        eto = pmFAO.calculate(
            tmean=tmean,
            delta=delta,
            es=es,
            ea=ea,
            psi_const=psi_const,
            rn=rn,
            u2=u2,
            g=g,
        )
        self.assertAlmostEqual(eto, 5.72, delta=0.01)

    def test_eto_with_daily_data(self):
        """ Testing Determination of ETo with daily data - Example 18 - pg.72"""
        ...