import unittest
from datetime import datetime

import numpy as np

from pet_equations.methods_parameters.astronomical_variables import (
    extra_terrestrial_radiation,
    relative_distance_earth_sun,
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
        self.assertEqual(self.DOY, 246)

    def test_relative_distance_earth_sun(self):
        self.assertAlmostEqual(relative_distance_earth_sun(self.DOY), 0.985, delta=0.1)

    def test_solar_declination(self):
        self.assertAlmostEqual(solar_declination(self.DOY), 0.12, delta=0.1)

    def test_sunset_hour_angle(self):
        latitude = np.deg2rad(self.LATITUDE_GRAUS)
        solar_dec = solar_declination(self.DOY)
        self.assertAlmostEqual(
            sunset_hour_angle(latitude, solar_dec),
            1.527,
            delta=0.1,
        )

    def test_extra_terrestrial_radiation(self):
        rel_dist_es = relative_distance_earth_sun(self.DOY)
        latitude = np.deg2rad(-20)
        solar_dec = solar_declination(self.DOY)
        sha = sunset_hour_angle(latitude, solar_dec)
        self.assertAlmostEqual(
            extra_terrestrial_radiation(rel_dist_es, sha, latitude, solar_dec),
            32.2,
            delta=0.1,
        )
