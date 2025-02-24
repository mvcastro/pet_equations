import unittest

import numpy as np

import pet_equations.methods_parameters.meteorological_variables as atm

DOY = np.array(246)


class TestMeteorologicalVariables(unittest.TestCase):
    """
    Test realized from the document: Allen et al. (1998). Crop evapotranspiration
    (guidelines for computing crop water requirements). FAO Irrigation and Drainage Paper 56.
    United Nations, Rome.
    """

    def test_atmospheric_pressure(self):
        """Testing Atmospheric Pressure Equation (Patm) - pg 31"""
        elevation = np.array(1800.0)
        atm_p = atm.atmospheric_pressure(altitude=elevation)
        self.assertAlmostEqual(atm_p, 81.8, delta=0.05)

    def test_psychrometric_constant(self):
        """Testing Psychrometric Constant Equation (γ) - pg 32"""
        elevation = np.array(1800.0)
        atm_p = atm.atmospheric_pressure(altitude=elevation)
        psy_const = atm.psychrometric_constant(atmospheric_pressure=atm_p)
        self.assertAlmostEqual(psy_const, 0.054, delta=0.01)

    def test_tmax_saturation_vapour_pressure_curve(self):
        """Testing Saturation Vapour Pressure with Tmax (∆) - pg 36"""
        tmax = np.array(24.5)
        svp_delta = atm.saturation_vapor_pressure(temperature=tmax)
        self.assertAlmostEqual(svp_delta, 3.075, delta=0.01)

    def test_tmin_saturation_vapour_pressure_curve(self):
        """Testing Saturation Vapour Pressure with Tmin (∆) - pg 36"""
        tmin = np.array(15.0)
        svp_delta = atm.saturation_vapor_pressure(temperature=tmin)
        self.assertAlmostEqual(svp_delta, 1.705, delta=0.01)

    def test_mean_saturation_vapour_pressure_curve(self):
        """Testing Mean Saturation Vapour Pressure Curve with Tmin and Tmax (∆) - pg 36"""
        tmax = np.array(24.5)
        tmin = np.array(15.0)
        svp_delta = atm.mean_saturation_vapor_pressure(tmin=tmin, tmax=tmax)
        self.assertAlmostEqual(svp_delta, 2.39, delta=0.01)

    def test_slope_saturation_vapour_pressure_curve(self):
        """Testing Slope Saturation Vapour Pressure Curve (∆) - pg 216 Table 2.4 for T=25.0C"""
        svp_delta = atm.slope_saturation_vapour_pressure_curve(
            temperature=np.array(25.0)
        )
        self.assertAlmostEqual(svp_delta, 0.189, delta=0.01)

    def test_actual_vapour_pressure(self):
        """Testing Actual Vapour Pressure Equation (ea) - Example 5 - pg.39"""
        tmax = np.array(25.0)
        tmin = np.array(18.0)
        rhmean = np.array((82 + 54) / 2)
        es = atm.mean_saturation_vapor_pressure(tmin=tmin, tmax=tmax)
        ea = atm.actual_vapour_pressure(es=es, rhmean=rhmean)
        self.assertAlmostEqual(ea, 1.78, delta=0.01)

    def test_wind_speed_at_height_of_2_m(self):
        """Testing Adjusting wind speed data to standard height - Example 14 - pg.56"""

        # Measured wind speed of 3.2 m/s at 10 m above the soil surface
        u2 = atm.wind_speed_at_2m(uz=3.2, z=10.0)
        self.assertAlmostEqual(u2, 2.4, delta=0.01)
