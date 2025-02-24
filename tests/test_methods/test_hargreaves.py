"""Teste da equação de HArgreaves"""

import unittest
from datetime import datetime

import numpy as np
import pandas as pd

import pet_equations.methods_parameters.astronomical_variables as avars
import pet_equations.methods_parameters.radiation_variables as rvars
from pet_equations.methods import hargreaves


class TestHargreaves(unittest.TestCase):
    """Test realized from example 20, page 77 of Allen et al. (1998).
        Crop evapotranspiration (guidelines for computing crop water requirements.
        FAO Irrigation and Drainage Paper 56.
        United Nations, Rome.
    """
    latitude = np.array(45.72)
    date = datetime(2021, 7, 15)
    doy = np.array(date.timetuple().tm_yday)
    tmax = np.array(26.6)
    tmin = np.array(14.8)

    def test_extra_terrestrial_radiation(self):
        rel_dist_es = avars.inverse_relative_distance_earth_sun(doy=self.doy)
        solar_dec = avars.solar_declination(doy=self.doy)
        sha = avars.sunset_hour_angle(
            latitude=np.deg2rad(self.latitude),
            solar_dec=solar_dec
            )
        ra = rvars.extra_terrestrial_radiation(
            rel_dist_es=rel_dist_es,
            sha=sha,
            latitude=np.deg2rad(self.latitude),
            solar_dec=solar_dec
        )
        
        self.assertAlmostEqual(ra, 40.55, delta=0.1)

    def test_hargreaves(self):
        """Hargreaves - Testing the equation"""

        self.assertAlmostEqual(
            hargreaves.calculate(
                latitude=self.latitude,
                tmin=self.tmin,
                tmax=self.tmax,
                tmean=(self.tmin + self.tmax) / 2,
                doy=self.doy
            ),
            5.0,
            delta=0.1
        )

    def test_hargreaves_series(self):
        """Hargreaves - Testing the equation"""

        latitude = np.array(45)
        date_lst = [datetime(2021, 7, 15), datetime(2021, 7, 16)]
        doy = pd.Series(np.array([i.timetuple().tm_yday for i in date_lst]))
        tmax = pd.Series(np.array([26.6, 25]))
        tmin = pd.Series(np.array([14.8, 16]))

        try:
            hargreaves.calculate(
                latitude, tmin, tmax, (tmin + tmax) / 2, doy)
        except ValueError as error:
            self.fail(f"myFunc() raised ExceptionType unexpectedly! - {error}")

    def test_hargreaves_exeception(self):
        """Hargreaves - Testing raise exception for different array lenghts of the parameters"""

        latitude = 45
        doy = np.array([1, 2, 3])
        tmax = 26.6
        tmin = 14.8

        with self.assertRaises(ValueError):
            hargreaves.calculate(
                latitude, tmin, tmax,
                (tmin + tmax) / 2, doy
            )  
                (tmin + tmax) / 2, doy
            )  
