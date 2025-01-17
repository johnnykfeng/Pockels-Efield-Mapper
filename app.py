import streamlit as st
import numpy as np
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
    pctl_color_range=False,
):
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

with st.sidebar:
    crop_range_x = st.slider("Crop range x", min_value=0, max_value=640, value=[5, 635])
    crop_range_y = st.slider(
        "Crop range y", min_value=0, max_value=512, value=[190, 320]
    )
    min_pctl = st.slider(
        "Lower range percentile", min_value=0.0, max_value=10.0, value=1.0
    )
    max_pctl = st.slider(
        "Upper range percentile", min_value=90.0, max_value=100.0, value=99.5
    )
    global_range_pctl = [min_pctl, max_pctl]
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
    fix_dead_pixels = st.checkbox("Fix dead pixels", value=False)
    save_E_field_data = st.checkbox("Save E-field data", value=True)
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


st.title("Pockels Image Analyzer")
st.write("Upload a PNG image to analyze its data.")

calib_img_arrays = {}
# File uploader widget
uploaded_calib_files = st.file_uploader(
    "Upload Calibration Images", type=["png"], accept_multiple_files=True
)

if uploaded_calib_files:
    for uploaded_calib_file in uploaded_calib_files:
        # Read image file
        img_array = png_to_array(uploaded_calib_file)
        img_array = crop_image(img_array, crop_range_x, crop_range_y)
        dead_pixels = find_dead_pixels(img_array, threshold=dead_pixel_threshold)
        if len(dead_pixels) > 0:
            st.write(f"Number of dead pixels: {len(dead_pixels)}")
            st.write(f"Dead pixels: {dead_pixels}")
            if fix_dead_pixels:
                img_array = impute_bad_pixels(img_array, dead_pixels)

        calib_img_arrays[remove_extension(uploaded_calib_file.name)] = img_array

        with st.expander(f"Plot {uploaded_calib_file.name}"):
            calib_fig = plot_image_with_color_slider(
                img_array, f"{uploaded_calib_file.name}", color_map, global_range_pctl
            )
            st.plotly_chart(calib_fig)
            histogram = plot_histogram(img_array, f"{uploaded_calib_file.name}")
            col1, col2 = st.columns([4, 1])
            with col1:
                st.plotly_chart(histogram)
            with col2:
                st.write(f"Mean: {image_array_statistics(img_array)[0]:.2f}")
                st.write(f"Std: {image_array_statistics(img_array)[1]:.2f}")
                st.write(f"Min: {image_array_statistics(img_array)[2]:.2f}")
                st.write(f"Max: {image_array_statistics(img_array)[3]:.2f}")

if calib_img_arrays:
    st.write(calib_img_arrays.keys())
    calib_parallel_on = calib_img_arrays["calib_parallel_on"]
    calib_parallel_off = calib_img_arrays["calib_parallel_off"]
    calib_cross_on = calib_img_arrays["calib_cross_on"]

uploaded_data_files = st.file_uploader(
    "Upload Data Images", type=["png"], accept_multiple_files=True
)

save_dir = r"C:\Users\10552\OneDrive - Redlen Technologies\Code\Pockels-Efield-Mapper\DATA\pockels_run_2025-01-16_b\E-field_data"

if uploaded_data_files:
    for uploaded_data_file in uploaded_data_files:
        filename = remove_extension(uploaded_data_file.name)
        img_array = png_to_array(uploaded_data_file)
        img_array = crop_image(img_array, crop_range_x, crop_range_y)

        if calib_img_arrays:
            numerator = img_array - calib_img_arrays["calib_cross_on"]
            denominator = (
                calib_img_arrays["calib_parallel_on"]
                - calib_img_arrays["calib_parallel_off"]
            )
            # numerator = img_array
            # denominator = calib_img_arrays["calib_parallel_on"]
            denominator = remove_low_value_pixels(denominator)
            T_array = numerator / denominator

        with st.expander(f"Plot {filename}"):
            img_fig = plot_image_with_color_slider(
                img_array, filename, color_map, global_range_pctl
            )
            st.plotly_chart(img_fig)
            if calib_img_arrays:
                # plot_image_with_color_slider(denominator, f"Denominator_{uploaded_data_file.name}", color_map, global_range_pctl)
                # plot_image_with_color_slider(numerator, f"Numerator_{uploaded_data_file.name}", color_map, global_range_pctl)

                do_cap_array = st.checkbox(
                    "Normalize transmission (0<T<1)",
                    value=True,
                    key=f"do_cap_array_{filename}",
                )
                if do_cap_array:
                    T_array = cap_array(T_array, 0, 1.0)
                T_fig = plot_image_with_color_slider(
                    T_array,
                    f"Transmission_{filename}",
                    color_map,
                    global_range_pctl,
                    pctl_color_range=True,
                )
                st.plotly_chart(T_fig)

                do_E_field = st.checkbox(
                    "Calculate E-field",
                    value=True,
                    key=f"do_E_field_{filename}",
                )
                if do_E_field and do_cap_array:
                    E_field_array = E_field_from_T(T_array, alpha)
                    E_fig = plot_image_with_color_slider(
                        E_field_array,
                        f"E-field_{filename}",
                        color_map,
                        global_range_pctl,
                        pctl_color_range=True,
                    )
                    st.plotly_chart(E_fig)

                    if save_E_field_data:
                        save_array_to_png(E_field_array, f"E-field_{filename}", save_dir=save_dir)
                        save_array_to_csv(E_field_array, f"E-field_{filename}", save_dir=save_dir)
                        save_plotly_figure(E_fig, f"E-field_{filename}", save_dir=save_dir)

