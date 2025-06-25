import cv2
import numpy as np
from scipy.signal import find_peaks

def remove_extension(file_name):
    return file_name.split(".")[0]

def get_metadata_from_filename(file_name):
    bias = file_name.split("_")[1]
    xray_flux = file_name.split("_")[3]
    # led_flux = file_name.split("_")[5]
    return bias, xray_flux

# def space_charge_density_from_E_field(E_field_A, E_field_C, distance: float):
#     """
#     Calculate the space charge density from the E-field at two points.
#     Args:
#         E_field_A: E-field near the anode
#         E_field_C: E-field near the cathode
#         distance: distance between the A and C points in meters
#     Returns:
#         rho: space charge density
#     """
#     slope = (E_field_C - E_field_A)/distance
#     epsilon_0 = 8.8541878128e-12 # C^2/(N*m^2)
#     epsilon_czt = 10.9*epsilon_0
#     e_coulomb = 1.60217663e-19 # C
#     rho = (-1)*slope*epsilon_czt*1e-6/(e_coulomb) # e/cm^3
#     return rho

