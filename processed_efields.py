import streamlit as st
import numpy as np
import os
import glob

from image_process import (
    png_to_array,
    crop_image,
    find_dead_pixels,
    find_bright_pixels,
    impute_bad_pixels,
    cap_array,
    remove_low_value_pixels,
    save_array_to_png,
    save_array_to_csv,
    csv_to_array,
)
from plotting_modules import (
    create_plotly_figure,
    plot_histogram,
    image_array_statistics,
    save_plotly_figure,
)
from pockels_math import alpha, E_ref, E_field_from_T

st.set_page_config(layout="wide")


def remove_extension(file_name):
    return file_name.split(".")[0]


# @st.cache_data
def plot_image_with_color_slider(
                                image_array,
                                title,
                                color_map,
                                global_range_pctl,
                                pctl_color_range=False):
    if pctl_color_range:
        col1, col2 = st.columns(2)
        with col1:
            min_pctl = st.number_input(
                "Min percentile",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                key=f"{title}_min_pctl",
            )
        with col2:
            max_pctl = st.number_input(
                "Max percentile",
                min_value=0.0,
                max_value=100.0,
                value=95.0,
                key=f"{title}_max_pctl",
            )
        if min_pctl < max_pctl:
            color_range = np.percentile(image_array, q=[min_pctl, max_pctl])
        else:
            color_range = [float(np.min(image_array)), float(np.max(image_array))]
    else:
        vmin, vmax = np.percentile(image_array, q=global_range_pctl)

        color_range = st.slider(
            "Color range",
            min_value=float(np.min(image_array)),
            max_value=float(np.max(image_array)),
            value=[vmin, vmax],
            key=f"{title}_color_range",
        )

    fig = create_plotly_figure(
        image_array, title=title, cmap=color_map, color_range=color_range
    )
    return fig


folder_path = r".\DATA\pockels_run_2025-01-16_b\E-field_data"
with st.expander("Folder path"):
    st.info(f"Folder path: {folder_path}")
# find all files with .csv extension inside folder_path
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

E_field_shape = csv_to_array(csv_files[0]).shape
print(f"E_field_shape: {E_field_shape}")
st.write(f"E_field_shape: {E_field_shape}")

with st.sidebar:

    # Color map
    color_map = st.selectbox(
        "Color map", options=["jet", "viridis", "magma", "plasma", "inferno", "cividis"]
    )
    invert_color_map = st.checkbox("Invert color map", value=False)
    if invert_color_map:
        color_map = color_map + "_r"

    dead_pixel_threshold = st.slider(
        "Dead pixel threshold", min_value=0, max_value=1000, value=100
    )
    crop_range_x = st.slider("Crop range x", min_value=0, max_value=E_field_shape[1], value=[13, 622])
    crop_range_y = st.slider("Crop range y", min_value=0, max_value=E_field_shape[0], value=[20, 109])
    
    fix_dead_pixels = st.checkbox("Fix dead pixels", value=False)
    with st.expander("Coefficients"):
        wavelength = st.number_input("Wavelength (nm)", value=1550.0)
        n0 = st.number_input("Refractive index", value=2.8)
        d = st.number_input("Thickness (mm)", value=2.0)
        r41 = st.number_input("Electro-optic coefficient r41 (1e-12 m/V)", value=5.5)
        alpha = alpha(wavelength, n0, d, r41)
        E_ref = E_ref(wavelength, n0, d, r41)
        st.write(f"Alpha: {alpha:.2e} m/V")
        st.write(f"E_ref: {E_ref:.2e} V/m")
        st.write(f"Alpha = pi/(2*E_ref): {np.pi / (2 * E_ref):.2e} m/V")
        st.caption(
            "Values and equations taken from: \n\nCola, A., Dominici, L., & Valletta, A. (2022). Optical Writing and Electro-Optic Imaging of Reversible Space Charges in Semi-Insulating CdTe Diodes. Sensors, 22(4). https://doi.org/10.3390/s22041579"
        )


def get_voltage_from_filename(filename):
    import re

    match = re.search(r"bias_(\d+)", filename)
    if match:
        return int(match.group(1))
    else:
        return None



st.write("CSV files found:")
E_fields = {}
voltages = []
for csv_file in csv_files:
    # st.write(csv_file)
    filename = os.path.basename(csv_file)
    filename_without_extension = remove_extension(filename)
    voltage = get_voltage_from_filename(filename_without_extension)
    voltages.append(voltage)
    E_field_array = csv_to_array(csv_file)
    E_fields[voltage] = E_field_array


ordered_voltages = sorted(voltages)
selected_voltage = st.select_slider("Select voltage", options=ordered_voltages)
E_field_array = E_fields[selected_voltage]
# E_field_array = np.log10(E_field_array)
E_field_cropped = crop_image(E_field_array, crop_range_x, crop_range_y)

E_field_fig = create_plotly_figure(E_field_array, 
                                   title=filename_without_extension, 
                                   cmap=color_map)

E_field_cropped_fig = create_plotly_figure(E_field_cropped, 
                                          title=filename_without_extension, 
                                          cmap=color_map)  

col1, col2 = st.columns(2)
with col1:
    min_range = st.number_input("Min range", value=0.0, step=2.0e5)
    # min_range = np.log10(min_range)
with col2:
    max_range = st.number_input("Max range", value=3.0e6, step=2.0e5)
    # max_range = np.log10(max_range)

E_field_fig.update_layout(coloraxis_colorbar=dict(
    title="E_z (V/m)",
    # tickvals=[min_range, max_range],
    # ticktext=[f"{min_range:.2e}", f"{max_range:.2e}"]
))
E_field_fig.update_coloraxes(cmin=min_range, cmax=max_range)

st.plotly_chart(E_field_fig)

E_field_cropped_fig.update_layout(coloraxis_colorbar=dict(
    title="E_z (V/m)",
    # tickvals=[min_range, max_range],
    # ticktext=[f"{min_range:.2e}", f"{max_range:.2e}"]
))
E_field_cropped_fig.update_coloraxes(cmin=min_range, cmax=max_range)

st.plotly_chart(E_field_cropped_fig)
