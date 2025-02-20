import unittest

import numpy as np

import pet_equations.methods_parameters.astronomical_variables as avars
import pet_equations.methods_parameters.radiation_variables as rvars


class TestRadiationVariables(unittest.TestCase):
    """
    Test realized from the document: Allen et al. (1998). Crop evapotranspiration
    (guidelines for computing crop water requirements). FAO Irrigation and Drainage Paper 56.
    United Nations, Rome.
    """
    DOY = np.array(246)
    LATITUDE_GRAUS = -20.0

    def test_extra_terrestrial_radiation(self):
        rel_dist_es = avars.relative_distance_earth_sun(self.DOY)
        latitude = np.deg2rad(self.LATITUDE_GRAUS)
        solar_dec = avars.solar_declination(self.DOY)
        sha = avars.sunset_hour_angle(latitude, solar_dec)
        self.assertAlmostEqual(
            rvars.extra_terrestrial_radiation(rel_dist_es, sha, latitude, solar_dec),
            32.2,
            delta=0.1,
        )
