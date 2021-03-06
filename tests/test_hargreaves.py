"""Teste da equação de HArgreaves"""

import unittest
from datetime import datetime

import numpy as np
import pandas as pd
from pet_equations import hargreaves # type: ignore


class TestHargreaves(unittest.TestCase):
    """Test realized from example 20, page 77 of Allen et al. (1998).
        Crop evapotranspiration (guidelines for computing crop water requirements.
        FAO Irrigation and Drainage Paper 56.
        United Nations, Rome.
    """

    def test_hargreaves(self):
        """Hargreaves - Testing the equation"""

        latitude = np.array(45)
        date = datetime(2021, 7, 15)
        doy = np.array(date.timetuple().tm_yday)
        tmax = np.array(26.6)
        tmin = np.array(14.8)

        self.assertAlmostEqual(
            hargreaves.calculate(
                latitude, tmin, tmax, (tmin + tmax) / 2, doy), 5.0, delta=0.1)  # type: ignore

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
            hargreaves.calculate(latitude, tmin, tmax,  # type: ignore
                                 (tmin + tmax) / 2, doy)  # type: ignore
