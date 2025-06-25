import numpy as np

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

def space_charge_density_from_slope(slope, px_to_meters):
    """
    Calculate the space charge density from the E-field at two points.
    Args:
        E_field_A: E-field near the anode
        E_field_C: E-field near the cathode
        distance: distance between the A and C points in meters
    Returns:
        rho: space charge density
    """
    slope = slope / px_to_meters
    epsilon_0 = 8.8541878128e-12 # C^2/(N*m^2)
    epsilon_czt = 10.9*epsilon_0
    e_coulomb = 1.60217663e-19 # C
    rho = (-1)*slope*epsilon_czt*1e-6/(e_coulomb) # e/cm^3
    return rho
