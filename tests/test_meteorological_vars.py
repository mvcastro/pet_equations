import unittest
import numpy as np
from meteorological_vars import (relative_distance_earth_sun, solar_declination,
                                 sunset_hour_angle, extra_terrestrial_radiation)

DOY = np.array(246)


class TestMeteorologicalVariables(unittest.TestCase):
    """Tests realized from example 8, page 47 of Allen et al. (1998).
        Crop evapotranspiration (guidelines for computing crop water requirements.
        FAO Irrigation and Drainage Paper 56.
        United Nations, Rome.
    """

    def test_relative_distance_earth_sun(self):
        self.assertAlmostEqual(
            relative_distance_earth_sun(DOY), 0.985, delta=0.1)  # type: ignore

    def test_solar_declination(self):
        self.assertAlmostEqual(solar_declination(
            246), 0.12, delta=0.1)  # type: ignore

    def test_sunset_hour_angle(self):
        latitude = np.deg2rad(-20)
        solar_dec = solar_declination(246)  # type: ignore
        self.assertAlmostEqual(sunset_hour_angle(latitude, solar_dec),  # type: ignore
                               1.527, delta=0.1)  # type: ignore

    def test_extra_terrestrial_radiation(self):
        rel_dist_es = relative_distance_earth_sun(DOY)
        latitude = np.deg2rad(-20)
        solar_dec = solar_declination(DOY)
        sha = sunset_hour_angle(latitude, solar_dec)
        self.assertAlmostEqual(
            extra_terrestrial_radiation(
                rel_dist_es, sha, latitude, solar_dec), 13.1, delta=0.1)  # type: ignore
