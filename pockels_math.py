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


