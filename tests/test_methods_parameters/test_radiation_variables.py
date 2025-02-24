import unittest
from datetime import datetime

import numpy as np

import pet_equations.methods_parameters.astronomical_variables as avars
import pet_equations.methods_parameters.radiation_variables as rvars




class TestRadiationVariables(unittest.TestCase):
    """
    Test realized from the document: Allen et al. (1998). Crop evapotranspiration
    (guidelines for computing crop water requirements). FAO Irrigation and Drainage Paper 56.
    United Nations, Rome.
    """

    def test_extra_terrestrial_radiation(self):
        """Testing Extaterrestrial Readiation (Ra) - Example 8 - pg.47"""
        date = datetime(2023, 9, 3)
        doy = date.timetuple().tm_yday
        latitude_graus = -20.0
        rel_dist_es = avars.inverse_relative_distance_earth_sun(doy)
        latitude = np.deg2rad(latitude_graus)
        solar_dec = avars.solar_declination(doy)
        sha = avars.sunset_hour_angle(latitude, solar_dec)
        ra = rvars.extra_terrestrial_radiation(rel_dist_es, sha, latitude, solar_dec)
        self.assertAlmostEqual(ra, 32.2, delta=0.1)

    def test_solar_radiation(self):
        """Testing Solar Readiation (Ra) - Example 10 - pg.50"""
        # duration of sunshine in May = 220 hours 
        actual_duration_of_sunshine = 220.0 / 31.0
        date = datetime(2023, 5, 15)
        doy = date.timetuple().tm_yday
        latitude_graus = -22.9
        rel_dist_es = avars.inverse_relative_distance_earth_sun(doy)
        latitude = np.deg2rad(latitude_graus)
        solar_dec = avars.solar_declination(doy)
        sha = avars.sunset_hour_angle(latitude, solar_dec)
        daylight_hours = avars.daylight_hours(sha=sha)
        
        # Extra-Terrestrial Radiation (Ra)
        ra = rvars.extra_terrestrial_radiation(rel_dist_es, sha, latitude, solar_dec)
        self.assertAlmostEqual(ra, 25.1, delta=0.1)
        
        # Solar or Shortwave Radiation (Rs)
        rs = rvars.solar_radiation(
            n=actual_duration_of_sunshine, N=daylight_hours, ra=ra
        )
        self.assertAlmostEqual(rs, 14.5, delta=0.1)

    def test_net_longwave_radiation(self):
        """Testing Net Longwave Readitation (Rnl) - Example 11 - pg.52"""
        
        # Vapour Pressure of 2.1 kPa
        ea = 2.1
        tmax = 25.1
        tmin = 19.1
        station_elevation = 0.0
        ra = 25.1
        rs = 14.5
        
        # Clear-sky solar radiation (Rso)
        rso = rvars.clear_sky_shortwave_radiation(altitude=station_elevation, ra=ra)
        self.assertAlmostEqual(rso, 18.8, delta=0.1)

        # Net Longwave Radiation (Rnl)
        rnl = rvars.net_longwave_radiation(
            tmax=tmax,
            tmin=tmin, 
            rs=rs,
            rso=rso,
            ea=ea
        )
        self.assertAlmostEqual(rnl, 3.5, delta=0.1)
        
    def test_net_radiation(self):
        """Testing Net Radiation - Example 12 - pg.53"""
        rs = 14.5
        rnl = 3.5
        # Net Solar or Net Shortwave Radiation (Rns)
        rns = rvars.net_shortwave_radiation(rs=rs)
        self.assertAlmostEqual(rns, 11.1, delta=0.1)
        
        # Net Radiation (Rn)
        rn = rns - rnl
        self.assertAlmostEqual(rn, 7.6, delta=0.1)
        
    def test_soil_heat_flux_for_monthly_periods(self):
        """Testing Soil Heat Flux for monthly periods - Example 13 - pg.55"""
        temp_march = 14.1
        temp_april = 16.1
        temp_may = 18.8
        
        g_april = rvars.soil_heat_flux(
            temp_iminus1=temp_march,
            temp_i=temp_april,
            temp_iplus1=temp_may,
            time_interval='monthly'
        )
        self.assertAlmostEqual(g_april, 0.33, delta=0.1)
        