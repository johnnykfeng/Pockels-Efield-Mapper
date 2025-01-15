import streamlit as st
import numpy as np
from png_analysis import png_to_array, crop_image, find_dead_pixels, impute_dead_pixels
from plotting_modules import create_plotly_figure
# import toml

# config = toml.load("config.toml")
# crop_range_x = config["crop_range_x"]
# crop_range_y = config["crop_range_y"]

st.set_page_config(layout="wide")

def remove_extension(file_name):
    return file_name.split(".")[0]

def plot_array(image_array, title, color_map, range_pctl):
    vmin, vmax = np.percentile(image_array, q = range_pctl)
    color_range = st.slider("Color range", 
                            min_value=float(np.min(image_array)), 
                            max_value=float(np.max(image_array)), 
                            value=[vmin, vmax])
    fig = create_plotly_figure(image_array, 
                                title=title, 
                                cmap=color_map, 
                                color_range=color_range)
    st.plotly_chart(fig)

with st.sidebar:
    crop_range_x = st.slider("Crop range x", min_value=0, max_value=640, value=[5, 635])
    crop_range_y = st.slider("Crop range y", min_value=0, max_value=512, value=[190, 320])
    min_pctl = st.slider("Lower range percentile", min_value=0.0, max_value=10.0, value=1.0)
    max_pctl = st.slider("Upper range percentile", min_value=90.0, max_value=100.0, value=99.5)
    range_pctl = [min_pctl, max_pctl]
    # Color map
    color_map = st.selectbox("Color map", options=["jet", "viridis", "magma", "plasma", "inferno", "cividis"])
    invert_color_map = st.checkbox("Invert color map", value=False)
    if invert_color_map:
        color_map = color_map + "_r"
    
    dead_pixel_threshold = st.slider("Dead pixel threshold", min_value=0, max_value=1000, value=100)
    fix_dead_pixels = st.checkbox("Fix dead pixels", value=False)


st.title("Pockels Image Analyzer")
st.write("Upload a PNG image to analyze its data.")

calib_img_arrays = {}
# File uploader widget
uploaded_calib_files = st.file_uploader("Upload Calibration Images", 
                                        type=["png"], 
                                        accept_multiple_files=True)

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
                img_array = impute_dead_pixels(img_array, dead_pixels)

        calib_img_arrays[remove_extension(uploaded_calib_file.name)] = img_array

        with st.expander(f"Plot {uploaded_calib_file.name}"):
            # Calculate color range
            vmin, vmax = np.percentile(img_array, q = range_pctl)
            color_range = st.slider("Color range", 
                                    min_value=float(np.min(img_array)), max_value=float(np.max(img_array)), 
                                    value=[vmin, vmax])
            # Convert to numpy array    
            fig = create_plotly_figure(img_array, 
                                    title=f"{uploaded_calib_file.name}", 
                                    cmap=color_map, 
                                    color_range=color_range)
            st.plotly_chart(fig)

if calib_img_arrays:
    st.write(calib_img_arrays.keys())
    calib_parallel_on = calib_img_arrays["calib_parallel_on"]
    calib_parallel_off = calib_img_arrays["calib_parallel_off"]
    calib_cross_on = calib_img_arrays["calib_cross_on"]

uploaded_data_files = st.file_uploader("Upload Data Images", 
                                       type=["png"], 
                                       accept_multiple_files=True)

if uploaded_data_files:
    for uploaded_data_file in uploaded_data_files:
        img_array = png_to_array(uploaded_data_file)
        img_array = crop_image(img_array, crop_range_x, crop_range_y)

        if calib_img_arrays:
            numerator = img_array - calib_cross_on
            denominator = calib_parallel_on - calib_parallel_off
            denominator[denominator < 1.0] = 1.0
            T_array = numerator / denominator
            # Clip transmission values to valid arcsin range
            # T_array[T_array > 1.0] = 0.99
            # T_array[T_array < -1.0] = -0.99
            # E_field = np.arcsin(T_array)
        
        with st.expander(f"Plot {uploaded_data_file.name}"):
            plot_array(img_array, f"{uploaded_data_file.name}", color_map, range_pctl)
            if calib_img_arrays:
                plot_array(T_array, f"{uploaded_data_file.name}", color_map, range_pctl)
            # # Calculate color range
            # vmin, vmax = np.percentile(array_for_plot, q = range_pctl)
            # color_range = st.slider("Color range", 
            #                         min_value=float(np.min(array_for_plot)), 
            #                         max_value=float(np.max(array_for_plot)), 
            #                         value=[vmin, vmax])
            # # Convert to numpy array    
            # fig = create_plotly_figure(array_for_plot, 
            #                         title=f"{uploaded_data_file.name}", 
            #                         cmap=color_map, 
            #                         color_range=color_range)
            # st.plotly_chart(fig)





