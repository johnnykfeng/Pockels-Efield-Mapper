import numpy as np

e_coulomb = 1.60217663e-19 # C
epsilon_0 = 8.8541878128e-12 # C^2/(N*m^2)
CONVERSION = {"e_cm3": 1e-6/e_coulomb, # C/m^3 to e/cm^3
              "e_m3": 1/e_coulomb} # C/m^3 to e/m^3

def alpha(wavelength, n0, d, r41):
    """
    Calculate the alpha coefficient for the electro-optic effect.
    """
    return np.sqrt(3) * np.pi * n0**3 * (d*1e-3) * (r41*1e-12) /(2* wavelength * 1e-9)

def E_ref(wavelength, n0, d, r41):
    """
    Calculate the reference electric field for the electro-optic effect.
    """
    return (wavelength * 1e-9)/(np.sqrt(3) * n0**3 * (d*1e-3) * (r41 * 1e-12))

def E_field_from_T(T_array, alpha):
    return np.sqrt(np.arcsin(T_array))/alpha


def space_charge_density_from_slope(slope_px, px_to_meters):
    """
    Calculate the space charge density from the slope of the E-field from the bias image.
    Args:
        slope_px: slope of the E-field from the bias image
        px_to_meters: conversion factor from pixels to meters
    Returns:
        rho: space charge density
    """
    slope = slope_px / px_to_meters
    epsilon_czt = 10.9*epsilon_0
    rho = (-1)*slope*epsilon_czt*CONVERSION["e_cm3"] # e/cm^3
    return rho
