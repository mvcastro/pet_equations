import unittest
from datetime import date

import numpy as np

import pet_equations.methods.penman_monteith_FAO as pmFAO
import pet_equations.methods_parameters.astronomical_variables as avars
import pet_equations.methods_parameters.meteorological_variables as mvars
import pet_equations.methods_parameters.radiation_variables as rvars


class TestPenmanMontheith(unittest.TestCase):
    """Test realized from the document: Circular Técnica:
    Roteiro de cálculo da evapotranspiração de referência pelo método de PenmanMonteith-FAO
    ISSN 1808-6810

    Local: Ilha Solteira, SP - Latitude 20º25'S
    Altitude (z) =  335 m
    Dia 15/10/2004
    Rn = 12.3 MJ/m2dia
    G = 0.6 MJ/m2dia
    Tmean = 25.6 °C
    Tmax = 32.3 °C
    Tmin = 22.3 °C
    UR = 81.6%
    U2 = 1.6 m/s
    Rs =  17.6 MJ/m2dia
    """

    LATITUDE = -20.416667
    ALTITUDE = 335.0
    DATE = date(year=2004, month=10, day=15)
    RN = 12.3
    G = 0.6
    TMEAN = 25.6
    TMAX = 32.3
    TMIN = 22.3
    UR = 81.6
    U2 = 1.6
    RS = 17.6

    def test_slope_saturation_vapour_pressure_curve(self):
        """Testing Slope Saturation Vapour Pressure Curve  (∆)"""
        svp_delta = mvars.slope_saturation_vapour_pressure_curve(
            temperature=np.array(self.TMEAN)
        )
        self.assertAlmostEqual(svp_delta, 0.195, delta=0.01)

    def test_atmospheric_pressure(self):
        """Testing Atmospheric Pressure Equation (Patm)"""
        atm_p = mvars.atmospheric_pressure(altitude=np.array(self.ALTITUDE))
        self.assertAlmostEqual(atm_p, 97.402, delta=0.01)

    def test_psychrometric_constant(self):
        """Testing Psychrometric Constant Equation (γ)"""
        atm_p = mvars.atmospheric_pressure(altitude=np.array(self.ALTITUDE))
        psy_const = mvars.psychrometric_constant(atmospheric_pressure=atm_p)
        self.assertAlmostEqual(psy_const, 0.065, delta=0.01)

    def test_saturation_vapor_pressure(self):
        """Testing Saturation Vapor Pressure Equation (es)"""
        es = mvars.saturation_vapor_pressure(temperature=np.array(self.TMEAN))
        self.assertAlmostEqual(es, 3.283, delta=0.01)

    def test_actual_vapour_pressure(self):
        """Testing Actual Vapour Pressure Equation (ea)"""
        es = mvars.saturation_vapor_pressure(temperature=np.array(self.TMEAN))
        ea = mvars.actual_vapour_pressure(es=es, rhmean=np.array(self.UR))
        self.assertAlmostEqual(ea, 2.679, delta=0.01)

    def test_net_shortwave_radiation(self):
        """Testing Net Shortwave Radiation Equation (Rns)"""
        rns = rvars.net_shortwave_radiation(rs=np.array(self.RS))
        self.assertAlmostEqual(rns, 13.55, delta=0.01)

    def test_sunset_hour_angle(self):
        """Testing Sunset Hour Angle Equation (ωs)"""
        doy = 288
        lat_rad = np.deg2rad(self.LATITUDE)
        solar_dec = avars.solar_declination(doy=np.array(doy))
        sha = avars.sunset_hour_angle(latitude=lat_rad, solar_dec=np.array(solar_dec))
        self.assertAlmostEqual(sha, 1.634, delta=0.01)

    def test_extra_terrestrial_radiation(self):
        """Testing Extra-terrestrial Radiation Equation (Ra)"""
        doy = 288
        lat_rad = np.deg2rad(self.LATITUDE)
        rel_dist_es = avars.inverse_relative_distance_earth_sun(doy=np.array(doy))
        solar_dec = avars.solar_declination(doy=np.array(doy))
        sha = avars.sunset_hour_angle(latitude=lat_rad, solar_dec=np.array(solar_dec))
        ra = rvars.extra_terrestrial_radiation(
            rel_dist_es=rel_dist_es,
            sha=sha,
            latitude=lat_rad,
            solar_dec=np.array(solar_dec),
        )
        self.assertAlmostEqual(ra, 38.565, delta=0.01)

    def test_clear_sky_shortwave_radiation(self):
        """Testing Clear Sky Shortwave Radiation Equation (Rso)"""
        doy = 288
        lat_rad = np.deg2rad(self.LATITUDE)
        rel_dist_es = avars.inverse_relative_distance_earth_sun(doy=np.array(doy))
        solar_dec = avars.solar_declination(doy=np.array(doy))
        sha = avars.sunset_hour_angle(latitude=lat_rad, solar_dec=np.array(solar_dec))
        ra = rvars.extra_terrestrial_radiation(
            rel_dist_es=rel_dist_es,
            sha=sha,
            latitude=lat_rad,
            solar_dec=np.array(solar_dec),
        )
        rso = rvars.clear_sky_shortwave_radiation(altitude=self.ALTITUDE, ra=ra)
        self.assertAlmostEqual(rso, 29.182, delta=0.01)

    def test_net_longwave_radiation(self):
        """Testing Net Longwave Radiation Equation (Rnl)"""
        es = mvars.saturation_vapor_pressure(temperature=np.array(self.TMEAN))
        ea = mvars.actual_vapour_pressure(es=es, rhmean=np.array(self.UR))

        doy = 288
        lat_rad = np.deg2rad(self.LATITUDE)
        rel_dist_es = avars.inverse_relative_distance_earth_sun(doy=np.array(doy))
        solar_dec = avars.solar_declination(doy=np.array(doy))
        sha = avars.sunset_hour_angle(latitude=lat_rad, solar_dec=np.array(solar_dec))
        ra = rvars.extra_terrestrial_radiation(
            rel_dist_es=rel_dist_es,
            sha=sha,
            latitude=lat_rad,
            solar_dec=np.array(solar_dec),
        )
        rso = rvars.clear_sky_shortwave_radiation(altitude=self.ALTITUDE, ra=ra)

        rnl = rvars.net_longwave_radiation(
            tmax=np.array(self.TMAX),
            tmin=np.array(self.TMIN),
            rs=np.array(self.RS),
            rso=rso,
            ea=ea,
        )
        self.assertAlmostEqual(rnl, 2.06, delta=0.01)
        
    def test_penman_monteith_fao_equation(self):
        
        atm_p = mvars.atmospheric_pressure(altitude=np.array(self.ALTITUDE))
        psy_const = mvars.psychrometric_constant(atmospheric_pressure=atm_p)
        es = mvars.saturation_vapor_pressure(temperature=np.array(self.TMEAN))
        ea = mvars.actual_vapour_pressure(es=es, rhmean=np.array(self.UR))
        svp_delta = mvars.slope_saturation_vapour_pressure_curve(
            temperature=np.array(self.TMEAN)
        )

        eto = pmFAO.calculate(
            tmean=np.array(self.TMEAN),
            delta=svp_delta,
            es=es,
            ea=ea,
            psi_const=psy_const,
            rn=np.array(self.RN),
            u2=np.array(self.U2),
            g=np.array(self.G),
        )
        self.assertAlmostEqual(eto, 3.79, delta=0.01)
