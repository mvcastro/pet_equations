import unittest
from datetime import datetime

import numpy as np

from pet_equations.methods_parameters.astronomical_variables import (
    daylight_hours,
    inverse_relative_distance_earth_sun,
    solar_declination,
    sunset_hour_angle,
)

DATE = datetime(2023, 9, 3)


class TestAstronomicalVariables(unittest.TestCase):
    """Tests realized from example 8, page 47 of Allen et al. (1998).
    Crop evapotranspiration (guidelines for computing crop water requirements.
    FAO Irrigation and Drainage Paper 56.
    United Nations, Rome.
    """

    DOY = DATE.timetuple().tm_yday
    LATITUDE_GRAUS = -20.0

    def test_day_of_year(self):
        """Testing Day of Year from a date - Example 8 - pg 47"""
        self.assertEqual(self.DOY, 246)

    def test_relative_distance_earth_sun(self):
        """Testing Inverse Relative Distance Earth-Sun (dr) - Example 8 - pg 47"""
        self.assertAlmostEqual(
            inverse_relative_distance_earth_sun(self.DOY), 0.985, delta=0.1
        )

    def test_solar_declination(self):
        """Testing Solar Declination (δ) - Example 8 - pg 47"""
        self.assertAlmostEqual(solar_declination(self.DOY), 0.12, delta=0.1)

    def test_sunset_hour_angle(self):
        """Testing Sunset Hour Angle (ωs) - Example 8 - pg 47"""
        latitude = np.deg2rad(self.LATITUDE_GRAUS)
        solar_dec = solar_declination(self.DOY)
        self.assertAlmostEqual(
            sunset_hour_angle(latitude, solar_dec),
            1.527,
            delta=0.1,
        )

    def test_daylight_hours(self):
        """Testing Daylight Hours (N)"""
        latitude = np.deg2rad(self.LATITUDE_GRAUS)
        solar_dec = solar_declination(self.DOY)
        sha = sunset_hour_angle(latitude, solar_dec)
        self.assertAlmostEqual(
            daylight_hours(sha),
            11.7,
            delta=0.1,
        )