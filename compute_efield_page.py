# External imports
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io
import tomllib
from pathlib import Path
import pandas as pd

# Internal imports
from utils import remove_extension, get_metadata_from_filename
from modules.pockels_math import alpha, E_ref, E_field_from_T, space_charge_density_from_slope
from modules.image_process import (
    png_to_array,
    crop_image,
    impute_bad_pixels,
    cap_array,
    find_bad_pixels,
    find_sensor_edges,
    plot_edge_detection_pipeline
)
from modules.plotting_modules import (
    plot_histogram,
    image_array_statistics,
    colored_pockels_images_matplotlib,
    heatmap_plot_with_bounding_box

)

st.set_page_config(layout="wide")

st.title("Pockels Image Analyzer")
st.write("Upload a PNG image to analyze its data.")


config_select = st.radio("CZT Configuration", options=["XMED", "CZT-10mm"], horizontal=True)

if config_select == "XMED":
    with open("config/XMED.toml", "rb") as f:
        config = tomllib.load(f)
elif config_select == "CZT-10mm":
    with open("config/CZT_10mm.toml", "rb") as f:
        config = tomllib.load(f)
        
if "figure_Efield_profile" not in st.session_state:
    st.session_state.figure_Efield_profile = None
if "mat_fig" not in st.session_state:
    st.session_state.mat_fig = None
if "bounding_box" not in st.session_state:
    st.session_state.bounding_box = None
if "px_to_meters" not in st.session_state:
    st.session_state.px_to_meters = None

@st.cache_data
def find_sensor_edges_cached(
    image, edge_threshold1=50, edge_threshold2=100, mean_threshold=6.0):
    return find_sensor_edges(image, 
                             edge_threshold1=edge_threshold1, 
                             edge_threshold2=edge_threshold2, 
                             mean_threshold=mean_threshold)


with st.sidebar: # Figure parameters
    sensor_id = st.text_input("Sensor ID", value="")
    st.subheader("Matplotlib Figure Settings")
    fig_height = st.slider("Figure height", min_value=100, max_value=1000, value=450)
    col1, col2 = st.columns(2)  
    with col1:
        matplot_axis_label_size = st.slider("Axis label size", min_value=1, max_value=20, value=10)
        matplot_tick_label_size = st.slider("Tick label size", min_value=1, max_value=20, value=8)
        color_map = st.selectbox(
            "Color map Raw Image", options=["jet", "viridis", "magma", "plasma", "inferno", "cividis"],
            index=1)
        min_pctl = st.number_input(
            "Lower range percentile", min_value=0.0, max_value=50.0, value=1.0)
        invert_color_map = st.checkbox("Invert color map", value=False)
        if invert_color_map:
            color_map = color_map + "_r"
    with col2:
        matplot_figure_width = st.slider("Figure width", min_value=1, max_value=20, value=6)
        matplot_figure_height = st.slider("Figure height", min_value=1, max_value=20, value=4)
        color_map_E_field = st.selectbox(
            "Color map E-field", options=["jet", "viridis", "magma", "plasma", "inferno", "cividis"],
            index=0)
        max_pctl = st.number_input(
            "Upper range percentile", min_value=50.0, max_value=100.0, value=99.0)
            
    global_range_pctl = [min_pctl, max_pctl]
    # Color map

    dead_pixel_threshold = st.slider("Dead pixel threshold", min_value=0, max_value=1000, value=100)
    hot_pixel_threshold = st.slider("Hot pixel threshold", min_value=10e3, max_value=100e3, value=20e3)
    fix_bad_pixels = st.checkbox("Fix bad pixels", value=False)

with st.sidebar: # Pockels parameters
    wavelength = st.number_input("Wavelength (nm)", value=config["wavelength"])
    n0 = st.number_input("Refractive index", value=config["n0"])
    d = st.number_input("Path Length (mm)", value=config["path_length"])
    r41 = st.number_input("Electro-optic coefficient r41 (1e-12 m/V)", value=config["r41"])
    alpha = alpha(wavelength, n0, d, r41)
    E_ref = E_ref(wavelength, n0, d, r41)
    st.write(f"Alpha: {alpha:.2e} m/V")
    st.write(f"E_ref: {E_ref:.2e} V/m")
    st.write(f"Alpha = pi/(2*E_ref): {np.pi / (2 * E_ref):.2e} m/V")
    st.caption(
        "Values and equations taken from: \n\nCola, A., Dominici, L., & Valletta, A. (2022). Optical Writing and Electro-Optic Imaging of Reversible Space Charges in Semi-Insulating CdTe Diodes. Sensors, 22(4). https://doi.org/10.3390/s22041579"
    )

with st.expander("Cropping and Boundary Selection"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        crop_range_left = st.number_input("Crop range left", min_value=0, max_value=639, value=0)
    with col2:
        crop_range_right = st.number_input("Crop range right", min_value=0, max_value=639, value=639)
    with col3:
        crop_range_top = st.number_input("Crop range top", min_value=0, max_value=511, value=0)
    with col4:
        crop_range_bottom = st.number_input("Crop range bottom", min_value=0, max_value=511, value=511)
    crop_range_x = [crop_range_left, crop_range_right]
    crop_range_y = [crop_range_top, crop_range_bottom]
    size_x = crop_range_right - crop_range_left
    size_y = crop_range_bottom - crop_range_top
    with col1:
        left_border = st.number_input("Left border", min_value=0, max_value=size_x, value=config["box_left"])
    with col2:
        right_border = st.number_input("Right border", min_value=0, max_value=size_x, value=config["box_right"])
    with col3:
        top_border = st.number_input("Top border", min_value=0, max_value=size_y, value=config["box_top"])
    with col4:
        bottom_border = st.number_input("Bottom border", min_value=0, max_value=size_y, value=config["box_bottom"])
    st.session_state.bounding_box = [left_border, top_border, right_border, bottom_border]
        
    apply_bounding_box = st.checkbox("Apply manual bounding box", value=True)

data_source = st.radio("Data source", 
                       options=["Data Uploader", "Sample Data"], 
                       horizontal=True,
                       index=1)
if data_source == "Data Uploader":
    col1, col2 = st.columns(2)
    with col1:
        uploaded_calib_files = st.file_uploader(
            "Upload Calibration Images", type=["png"], accept_multiple_files=True)
    with col2:
        uploaded_data_files = st.file_uploader(
            "Upload Bias Images", type=["png"], accept_multiple_files=True)
elif data_source == "Sample Data":
    calib_data_folder = Path("SAMPLE_DATA/LED_data/calib")
    bias_data_folder = Path("SAMPLE_DATA/LED_data/bias")
    
    # Check if sample data directories exist (they won't in deployed environment)
    if calib_data_folder.exists() and bias_data_folder.exists():
        uploaded_calib_files = list(calib_data_folder.glob("*.png"))
        uploaded_data_files = list(bias_data_folder.glob("*.png"))
        uploaded_data_files.sort()
    else:
        st.warning("⚠️ Sample data is not available.")
        uploaded_calib_files = []
        uploaded_data_files = []


calib_img_arrays = {}

#######################
### CALIBRATION PROCESSING ###
#######################

if uploaded_calib_files:
    all_bad_pixels = []
    for uploaded_calib_file in uploaded_calib_files:
        # Read image file
        img_array = png_to_array(uploaded_calib_file)
        img_array = crop_image(img_array, crop_range_x, crop_range_y)
        
        if uploaded_calib_file.name == "calib_cross_on.png" or uploaded_calib_file.name == "calib_cross_off.png":
            bad_pixels = find_bad_pixels(img_array, lower_threshold=dead_pixel_threshold, upper_threshold=hot_pixel_threshold)
        else:
            bad_pixels = find_bad_pixels(img_array, lower_threshold=dead_pixel_threshold, upper_threshold=65e3)
        if len(bad_pixels) > 0:
            st.write(f"Number of bad pixels: {len(bad_pixels)}")
            st.write(f"Bad pixels: {bad_pixels}")
            all_bad_pixels.extend(bad_pixels)
            if fix_bad_pixels:
                img_array = impute_bad_pixels(img_array, bad_pixels)
            

        calib_img_arrays[remove_extension(uploaded_calib_file.name)] = img_array

        with st.expander(f"Plot {uploaded_calib_file.name}"):
            vmin, vmax = np.percentile(img_array, q=global_range_pctl)
            color_range = st.slider("Color range", min_value=0.0, max_value=65500.0, value=[vmin, vmax])
            calib_fig = heatmap_plot_with_bounding_box(
                img_array, f"{uploaded_calib_file.name}", color_map, color_range, fig_height=fig_height
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

    if len(all_bad_pixels) > 0:
        st.sidebar.write(f"Number of bad pixels: {len(all_bad_pixels)}")
        st.sidebar.write(f"Bad pixels: {all_bad_pixels}")

if calib_img_arrays:
    try:
        # st.write(calib_img_arrays.keys())
        calib_parallel_on = calib_img_arrays["calib_parallel_on"]
        calib_parallel_off = calib_img_arrays["calib_parallel_off"]
        calib_cross_on = calib_img_arrays["calib_cross_on"]
        calib_parallel_minus_cross = np.subtract(calib_img_arrays["calib_parallel_on"], calib_img_arrays["calib_cross_on"])
        if st.checkbox("Calculate Parallel minus Cross", value=False):
            with st.expander("Parallel minus Cross"):
                vmin, vmax = np.percentile(calib_parallel_minus_cross, q=global_range_pctl)
                color_range = st.slider("Color range", min_value=0.0, max_value=65500.0, value=[vmin, vmax])
                st.plotly_chart(heatmap_plot_with_bounding_box(
                    calib_parallel_minus_cross, "Parallel minus Cross", color_map, color_range, 
                    fig_height=fig_height))
    except Exception as e:
        print(f"Error: {e}")
        st.warning(f"Error: {e}")
        st.warning("Not all calibration images are uploaded")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        apply_edge_detection = st.checkbox("Apply edge detection", value=True)
        top_margin = st.number_input("Top margin", min_value=-50, max_value=50, value=0)
    with col2:
        edge_threshold1 = st.number_input("Edge threshold 1", min_value=0, max_value=1000, value=80)
        bottom_margin = st.number_input("Bottom margin", min_value=-50, max_value=50, value=0)
    with col3:
        edge_threshold2 = st.number_input("Edge threshold 2", min_value=0, max_value=1000, value=100)
        left_margin = st.number_input("Left margin", min_value=-50, max_value=50, value=15)
    with col4:
        mean_threshold = st.number_input("Mean threshold", min_value=0.0, max_value=50.0, value=8.0)
        right_margin = st.number_input("Right margin", min_value=-50, max_value=50, value=-15)
    if apply_edge_detection:
        top_edge, bottom_edge, left_edge, right_edge, canny_edges, horizontal_edge_strength, vertical_edge_strength = \
            find_sensor_edges_cached(calib_img_arrays["calib_parallel_on"], 
                                    edge_threshold1=edge_threshold1, 
                                    edge_threshold2=edge_threshold2, 
                                    mean_threshold=mean_threshold)
        fig = plot_edge_detection_pipeline(calib_img_arrays["calib_parallel_on"], 
                                           top_edge, bottom_edge, left_edge, right_edge, 
                                           canny_edges, 
                                           horizontal_edge_strength, 
                                           vertical_edge_strength, 
                                           mean_threshold,
                                           top_margin, bottom_margin, left_margin, right_margin)
        with st.expander("Edge Detection"):
            bounding_box = [left_edge + left_margin, 
                            top_edge + top_margin, 
                            right_edge + right_margin, 
                            bottom_edge + bottom_margin]
            st.session_state.bounding_box = bounding_box
            st.session_state.px_to_meters = (2e-3)/abs(bottom_edge - top_edge)
            st.pyplot(fig)
            st.write({"left_edge": left_edge, "top_edge": top_edge, "right_edge": right_edge, "bottom_edge": bottom_edge})
            st.write({"bounding_box": bounding_box})
            st.write(f"px_to_meters: {st.session_state.px_to_meters}")


st.divider()
###########################################
### DATA PROCESSING ######################
###########################################
raw_image_plotly_figures = {}
E_field_arrays = {}
E_field_plotly_figures = {}
row_avg_E_field_arrays = {}


col1, col2, col3, col4 = st.columns(4)
with col1:
    Emin = st.number_input("Minimum E-field (kV/m)", min_value=0.0, max_value=10000.0, value=0.0)*1e3
    Emax = st.number_input("Maximum E-field (kV/m)", min_value=0.0, max_value=10000.0, value=800.0)*1e3
with col2:
    show_raw_image = st.checkbox("Show raw image", value=True, key=f"show_raw_image")
    show_numerator_denominator = st.checkbox("Show numerator and denominator", value=False, key=f"show_numerator_denominator")
    perform_T_normalization = st.checkbox("Normalize transmission (0<T<1)", value=True, key=f"normalize_transmission")
    show_transmission_image = st.checkbox("Show transmission image", value=False, key=f"show_transmission_image")
with col3:
    calculate_E_field = st.checkbox("Calculate E-field", value=True, key=f"do_E_field")
    calculate_row_avg_E_field = st.checkbox("Calculate row-wise average E-field", value=True, key="row_avg_E_field")
    show_Efield_profile = st.checkbox("Show E-field profile per image", value=False, key="show_Efield_profile")
with col4:
    cathode_margin = st.number_input("Cathode margin (pixels)", min_value=-20, max_value=20, value=0)
    anode_margin = st.number_input("Anode margin (pixels)", min_value=-20, max_value=20, value=0)

with st.sidebar:
    def format_func(option):
        mapping = {
            "norm_1": "I_bias/I_para",
            "norm_2": "(I_bias-I_cross)/(I_para)",
            "norm_3": "(I_bias-I_cross)/(I_para-I_off)"
        }
        return mapping[option]
    normalization_method = st.radio("Normalization method", 
                                    options=["norm_1", 
                                                "norm_2", 
                                                "norm_3"],
                                    format_func=format_func, index = 2)
    st.caption(f"Selected normalization: {normalization_method}")

if uploaded_data_files:
    for uploaded_data_file in uploaded_data_files:
        filename = remove_extension(uploaded_data_file.name)
        # bias, xray_flux = get_metadata_from_filename(filename)
        # st.write(f"Bias: {bias}, Xray flux: {xray_flux}")
        img_array = png_to_array(uploaded_data_file, dtype=np.float32)
        # img_array = img_array.astype(float)
        img_array = crop_image(img_array, crop_range_x, crop_range_y)

        if calib_img_arrays:
            if normalization_method == "norm_1":
                numerator = img_array
                denominator = calib_img_arrays["calib_parallel_on"]
            elif normalization_method == "norm_2":
                numerator = img_array - calib_img_arrays["calib_cross_on"]
                denominator = calib_img_arrays["calib_parallel_on"]
            elif normalization_method == "norm_3":
                numerator = img_array - calib_img_arrays["calib_cross_on"]
                denominator = calib_img_arrays["calib_parallel_on"] - calib_img_arrays["calib_parallel_off"]
            
            # numerator[numerator<0] = 0
            denominator[denominator<=0] = 1.0
            T_array = numerator / denominator
            

        with st.expander(f"Plot {filename}"):
            
            vmin, vmax = np.percentile(img_array, q=global_range_pctl)
            color_range = st.slider("Color range", min_value=0.0, max_value=65500.0, value=[vmin, vmax],
                                    key=f"raw_image_color_range_{filename}")
            img_fig = heatmap_plot_with_bounding_box(
                img_array, f"{uploaded_data_file.name}", color_map, color_range, 
                fig_height=fig_height, bounding_box=st.session_state.bounding_box
            )
   
            raw_image_plotly_figures[filename] = img_fig
            if show_raw_image:
                st.plotly_chart(img_fig)
            if calib_img_arrays:
                if show_numerator_denominator:
                    color_range = st.slider("Color range", min_value=0.0, max_value=65500.0, value=[vmin, vmax],
                                            key=f"numerator_color_range_{filename}")
                    numerator_heatmap = heatmap_plot_with_bounding_box(
                        numerator, f"Numerator_{uploaded_data_file.name}", color_map, color_range, 
                        fig_height=fig_height, bounding_box=st.session_state.bounding_box)
                    denominator_heatmap = heatmap_plot_with_bounding_box(
                        denominator, f"Denominator_{uploaded_data_file.name}", color_map, color_range, 
                        fig_height=fig_height, bounding_box=st.session_state.bounding_box)
                    st.plotly_chart(numerator_heatmap)
                    st.plotly_chart(denominator_heatmap)

                if perform_T_normalization:
                    T_array = cap_array(T_array, 1e-3, 1.0)
                if fix_bad_pixels:
                    T_array = impute_bad_pixels(T_array, all_bad_pixels)
                if show_transmission_image:
                    vmin, vmax = np.percentile(T_array, q=global_range_pctl)
                    color_range = st.slider("Color range", min_value=0.0, max_value=1.0, value=[vmin, vmax],
                                            key=f"transmission_color_range_{filename}")
                    T_fig = heatmap_plot_with_bounding_box(
                        T_array,
                        f"Transmission_{filename}",
                        color_map,
                        color_range,
                        fig_height=fig_height,
                        bounding_box=st.session_state.bounding_box,
                    )
                    st.plotly_chart(T_fig)


                if calculate_E_field and perform_T_normalization:
                    E_field_array = E_field_from_T(T_array, alpha)
                    E_field_arrays[filename] = E_field_array
                    color_range = st.slider("Color range", min_value=0.0, max_value=Emax*1.5, value=[Emin, Emax],
                                            key=f"E_field_color_range_{filename}")
                    E_fig = heatmap_plot_with_bounding_box(
                        E_field_array,
                        f"E-field_{filename}",
                        color_map_E_field,
                        color_range,
                        fig_height=fig_height,
                        bounding_box=st.session_state.bounding_box,
                    )
                    E_fig.update_layout(
                        coloraxis_colorbar=dict(
                            title="E-field (V/m)"))
                    E_field_plotly_figures[f"{filename}"] = E_fig # save the plotly figure
                    st.plotly_chart(E_fig)

                    # Plot row-wise average of E-field within bounding box
                    if st.session_state.bounding_box and calculate_row_avg_E_field:
                        size_image = np.shape(E_field_array)
                        # Extract the region within bounding box
                        x0, y0, x1, y1 = st.session_state.bounding_box
                        # Ensure bounding box doesn't exceed image dimensions
                        x0 = max(0, min(x0, size_image[1]-1))
                        x1 = max(0, min(x1, size_image[1]-1))
                        y0 = max(0, min(y0, size_image[0]-1))+cathode_margin
                        y1 = max(0, min(y1, size_image[0]-1))+anode_margin
                        try:
                            E_field_roi = E_field_array[y0:y1, x0:x1]
                        except Exception as e:
                            print(f"Error: {e}")
                            st.warning(f"Error: {e}")
                            st.write("Error: Bounding box exceeds image dimensions")
                            st.write(f"Size of image: {size_image}")
                            st.write(f"Bounding box: {st.session_state.bounding_box}")
                            st.write(f"Adjusted bounding box: {x0}, {x1}, {y0}, {y1}")

                        # Calculate row-wise average
                        row_avg = np.mean(E_field_roi, axis=1)
                        row_indices = np.arange(y0, y1)
                        distance_mm = (row_indices - y0)*st.session_state.px_to_meters*1e3
                        row_avg_E_field_arrays[filename] = {"E_row_avg": row_avg, 
                                                            "row_indices": row_indices,
                                                            "distance_mm": distance_mm}
                        if show_Efield_profile:
                            # Create figure for row average plot
                            fig, ax = plt.subplots(figsize=(matplot_figure_width, matplot_figure_height))
                            # Use actual pixel indices from original image
                            ax.plot(row_indices, row_avg, '-')
                            # Add shaded margins of 5 pixels on left and right
                            margin_width = 5
                            ax.axvspan(y0, y0 + margin_width, color='orange', alpha=0.1)  # Left margin
                            ax.axvspan(y1 - margin_width, y1, color='green', alpha=0.1)  # Right margin
                            # Add text annotations for anode and cathode regions
                            ax.text(y0 + margin_width/2, ax.get_ylim()[0]*1.02, 'cathode', 
                                horizontalalignment='center', verticalalignment='bottom', rotation=90)
                            ax.text(y1 - margin_width/2, ax.get_ylim()[0]*1.02, 'anode',
                                horizontalalignment='center', verticalalignment='bottom', rotation=90)
                            ax.set_xlabel('Row position (pixels)', fontsize=matplot_axis_label_size)
                            ax.set_ylabel('Average Ez-field (V/m)', fontsize=matplot_axis_label_size)
                            ax.set_title(f'Row-wise Average E-field for {filename}', fontsize=matplot_axis_label_size)
                            ax.grid(True, alpha=0.3)
                            # Format y-axis ticks in scientific notation
                            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                            # Increase tick label font size
                            ax.tick_params(axis='both', which='major', labelsize=matplot_tick_label_size)
                            st.pyplot(fig)

with st.expander("Row-wise Average E-field", expanded=True):
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        auto_ylim = st.checkbox("Auto y-axis limits", value=False)
    with col2:
        include_long_text = st.checkbox("Include E-field Equations", value=False)
    with col3:
        ylim_min = st.number_input("E-field lower limit (V/m)", min_value=0.0, max_value=1e6, value=Emin)
    with col4:
        ylim_max = st.number_input("E-field upper limit (V/m)", min_value=0.0, value=Emax)
    with col5:
        edge_margin = st.number_input("Edge margin (pixels)", min_value=0, max_value=20, value=5)
    with col6:
        convert_to_mm = st.checkbox("Convert row indices to mm", value=True)
    if row_avg_E_field_arrays:
        st.session_state.figure_Efield_profile, ax = plt.subplots(figsize=(matplot_figure_width, matplot_figure_height))
        # Invert the color order by reversing the color array
        colors = plt.cm.jet(np.linspace(1, 0, len(row_avg_E_field_arrays)))
        # Reverse the order of items for plotting
        for i, (filename, row_avg_E_field_array) in enumerate(
            reversed(list(row_avg_E_field_arrays.items()))):
            if convert_to_mm:
                row_indices = row_avg_E_field_array['distance_mm']
            else:
                row_indices = row_avg_E_field_array['row_indices']
            ax.plot(row_indices, row_avg_E_field_array['E_row_avg'], '-',
                    label=f'{filename}', color=colors[i])
            
        
        long_text = (
            "$T = \\frac{I_{bias}-I_{cross}}{I_{parallel}-I_{off}} = \\sin^2(\\alpha E)$\n\n"
            "$T$ is the normalized transmission\n"
            "$E$ is the vertical component of the electric field\n"
            "$\\alpha=\\frac{\\sqrt{3}\\pi}{2}\\cdot\\frac{d n_o^3 r_{41}}{\\lambda}$ is the Pockels parameter\n\n"
            f"$d={d}$ mm is the optical path length\n"
            f"$r_{{41}}={r41} \\times 10^{{-12}}$ m/V is the electro-optic coefficient\n" 
            f"$n_o={n0}$ is the refractive index\n"
            f"$\\lambda={wavelength}$ nm is the wavelength of the light\n\n"
        )

        # Add text annotation to bottom right of E-field profile figure
        if st.session_state.figure_Efield_profile is not None and include_long_text:
            # Get figure dimensions
            fig = st.session_state.figure_Efield_profile
            fig_width, fig_height = fig.get_size_inches()
            # if include_long_text:
            # Add text in figure but outside axes, in bottom right
            fig.text(0.68, 0.1, long_text,
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    fontsize=matplot_tick_label_size*0.7,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                    transform=fig.transFigure)
        
        # Add shaded margins of 5 pixels on left and right
        if convert_to_mm:
            ax.axvspan(0, 0.2, color='orange', alpha=0.2, label='cathode')  # Left margin
            ax.axvspan(distance_mm[-1] - 0.2, distance_mm[-1], color='green', alpha=0.2, label='anode')  # Right margin
            ax.set_xlabel('Distance from cathode (mm)', fontsize=matplot_axis_label_size)
        else:
            ax.axvspan(y0, y0 + edge_margin, color='orange', alpha=0.2, label='cathode')  # Left margin
            ax.axvspan(y1 - edge_margin, y1, color='green', alpha=0.2, label='anode')  # Right margin

        if not auto_ylim:
            ax.set_ylim([ylim_min, ylim_max])  # Set minimum y value to 0 while keeping auto maximum
        ax.set_ylabel('Average E-field (V/m)', fontsize=matplot_axis_label_size)
        ax.set_title('Row-wise Average E-field', fontsize=matplot_axis_label_size)
        ax.grid(True, alpha=0.5)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.tick_params(axis='both', which='major', labelsize=matplot_tick_label_size)
        # Place legend outside the plot
        ax.legend(fontsize=matplot_tick_label_size, bbox_to_anchor=(1.05, 1), loc='upper left')
        # Adjust layout to make room for the legend
        plt.tight_layout()
        st.pyplot(st.session_state.figure_Efield_profile)
    else:
        st.warning("No row-wise average E-field data available")
    

    # col1, col2, col3, col4 = st.columns(4)
    # with col1:
        
    if row_avg_E_field_arrays:
        # Convert row-wise E-field data to pandas DataFrame
        df_list = []
        for filename, row_data in row_avg_E_field_arrays.items():
            # Calculate gradient on the raw numpy arrays
            row_indices = np.array(row_data['row_indices'])[edge_margin:-edge_margin]
            e_field_values = np.array(row_data['E_row_avg'])[edge_margin:-edge_margin]
            slope_values = np.gradient(e_field_values, row_indices)
            mean_slope = np.mean(slope_values)
            rho = space_charge_density_from_slope(mean_slope, st.session_state.px_to_meters)
            
            df = pd.DataFrame({
                # 'Row Indices': [row_indices],
                # 'E-field (V/m)': [e_field_values],
                # 'slope': [slope_values],
                'filename': [filename],
                'tag': [filename.split('_')[-1]],
                'pixel2meters': [f"{st.session_state.px_to_meters:.3e}"],
                'slope [dE/pixel]': [f"{mean_slope:.3e}"],
                'rho [e/cm^3]': [f"{rho:.3e}"],
            })
            df_list.append(df)
        
        combined_df = pd.concat(df_list)
        st.dataframe(combined_df)
        st.download_button(
            label="Download Space Charge Density Data Table",
            data=combined_df.to_csv(index=False),
            file_name=f"{sensor_id}_space_charge_density_data.csv" \
                if sensor_id else "space_charge_density_data.csv",
            mime="text/csv"
        )

with st.expander("Efield Matplotlib Plots", expanded=True):
    if E_field_arrays:
        last_E_field = list(E_field_arrays.values())[-1] # get the last E-field array
        vmin, vmax = np.percentile(last_E_field, q=global_range_pctl)
        col1, col2, col3 = st.columns(3)
        with col1:
            color_range_radio = st.radio("Color range", options=["Auto", "Fixed"], index=0, key="color_range_radio")
        with col2:
            color_min, color_max = st.slider("Color range", min_value=0.0, max_value=Emax*1.5, 
                                             value=[vmin, vmax], step=10000.0, key="color_range_matplotlib")
        with col3:  
            box_color = st.color_picker("Box color", value="#FFFFFF", key="box_color")
        st.session_state.mat_fig = colored_pockels_images_matplotlib(E_field_arrays, 
                                                    color_range_radio, 
                                                    color_min, color_max, 
                                                    apply_bounding_box, 
                                                    st.session_state.bounding_box,
                                                    box_color)
        # Add colorbar title and set scientific notation
        for ax in st.session_state.mat_fig.axes:
            if ax.get_label() == '<colorbar>':
                ax.set_ylabel('Electric Field (V/m)', fontsize=matplot_axis_label_size)
                ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='sci', scilimits=(0,0))
        # Add text annotations for cathode and anode
        for ax in st.session_state.mat_fig.axes:
            ax.text(st.session_state.bounding_box[0], st.session_state.bounding_box[1]-5, 'cathode', 
                    color=box_color, fontsize=matplot_tick_label_size,
                    horizontalalignment='left')
            ax.text(st.session_state.bounding_box[0], st.session_state.bounding_box[3]+5, 'anode',
                    color=box_color, fontsize=matplot_tick_label_size, 
                    horizontalalignment='left')
        st.session_state.mat_fig.suptitle(f"E-field heatmaps {color_range_radio}-scale color range", fontsize=15, y=1.05)
        st.pyplot(st.session_state.mat_fig)

if st.session_state.mat_fig is not None and st.session_state.figure_Efield_profile is not None:
 
    # Create PDF buffer in memory
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        pdf.savefig(st.session_state.figure_Efield_profile, bbox_inches='tight')
        pdf.savefig(st.session_state.mat_fig, bbox_inches='tight')
    
    # Reset buffer position to start
    pdf_buffer.seek(0)
    
    
    with st.popover("Download PDF"):
        # sensor_id = st.text_input("Sensor_ID", value="Sensor_ID")
        st.download_button(
            label="Download E-field plots as PDF",
            data=pdf_buffer,
            file_name=f"{sensor_id}_Efield_plots.pdf" \
                if sensor_id else "Efield_plots.pdf",
            mime="application/pdf"
        )
