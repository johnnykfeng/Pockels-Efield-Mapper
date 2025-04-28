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
    find_bad_pixels
)
from plotting_modules import (
    create_plotly_figure,
    plot_histogram,
    image_array_statistics,
    save_plotly_figure,
)
from pockels_math import alpha, E_ref, E_field_from_T
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(layout="wide")

def remove_extension(file_name):
    return file_name.split(".")[0]

def heatmap_plot_with_bounding_box(
    image_array,
    title,
    color_map,
    color_range,
    fig_height=800,
    fig_width=800,
    bounding_box=None,
):
    
    fig = create_plotly_figure(
        image_array, title=title, cmap=color_map, color_range=color_range
    )
    if bounding_box:
        # Check if bounding box dimensions exceed image array dimensions
        if (bounding_box[2] > image_array.shape[1] or 
            bounding_box[3] > image_array.shape[0]):
            st.warning("Warning: Bounding box dimensions exceed image dimensions")
            bounding_box = [0, 0, image_array.shape[1], image_array.shape[0]]
        fig.add_shape(
            type="rect",
            x0=bounding_box[0],
            y0=bounding_box[1],
            x1=bounding_box[2],
            y1=bounding_box[3],
        line=dict(
            color="white",
            width=3,
            dash="dash"
            )
        )
        
    fig.update_layout(
        height=fig_height,
        width=fig_width
    )
    
    return fig

st.title("Pockels Image Analyzer")
st.write("Upload a PNG image to analyze its data.")

with st.sidebar:
    fig_height = st.slider("Figure height", min_value=100, max_value=1000, value=1000)
    
    col1, col2 = st.columns(2)  
    with col1:
        min_pctl = st.number_input(
            "Lower range percentile", min_value=0.0, max_value=50.0, value=1.0)
        color_map = st.selectbox(
            "Color map Raw Image", options=["jet", "viridis", "magma", "plasma", "inferno", "cividis"],
            index=1)
    with col2:
        max_pctl = st.number_input(
            "Upper range percentile", min_value=50.0, max_value=100.0, value=99.0)
        color_map_E_field = st.selectbox(
            "Color map E-field", options=["jet", "viridis", "magma", "plasma", "inferno", "cividis"],
            index=0)
    global_range_pctl = [min_pctl, max_pctl]
    # Color map
    invert_color_map = st.checkbox("Invert color map", value=False)
    if invert_color_map:
        color_map = color_map + "_r"

    dead_pixel_threshold = st.slider(
        "Dead pixel threshold", min_value=0, max_value=1000, value=100
    )
    hot_pixel_threshold = st.slider(
        "Hot pixel threshold", min_value=10e3, max_value=100e3, value=20e3
    )
    # fix_dead_pixels = st.checkbox("Fix dead pixels", value=False)
    fix_bad_pixels = st.checkbox("Fix bad pixels", value=False)
    
    save_E_field_data = st.checkbox("Save E-field data", value=False)

    wavelength = st.number_input("Wavelength (nm)", value=1550.0)
    n0 = st.number_input("Refractive index", value=2.8)
    d = st.number_input("Path Length (mm)", value=8.17)
    r41 = st.number_input("Electro-optic coefficient r41 (1e-12 m/V)", value=5.5)
    alpha = alpha(wavelength, n0, d, r41)
    E_ref = E_ref(wavelength, n0, d, r41)
    st.write(f"Alpha: {alpha:.2e} m/V")
    st.write(f"E_ref: {E_ref:.2e} V/m")
    st.write(f"Alpha = pi/(2*E_ref): {np.pi / (2 * E_ref):.2e} m/V")
    st.caption(
        "Values and equations taken from: \n\nCola, A., Dominici, L., & Valletta, A. (2022). Optical Writing and Electro-Optic Imaging of Reversible Space Charges in Semi-Insulating CdTe Diodes. Sensors, 22(4). https://doi.org/10.3390/s22041579"
    )

with st.expander("Image Processing"):
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
    apply_bounding_box = st.checkbox("Apply bounding box", value=True)
    if apply_bounding_box:
        with col1:
            left_border = st.number_input("Left border", min_value=0, max_value=size_x, value=0)
        with col2:
            right_border = st.number_input("Right border", min_value=0, max_value=size_x, value=size_x)
        with col3:
            top_border = st.number_input("Top border", min_value=0, max_value=size_y, value=0)
        with col4:
            bottom_border = st.number_input("Bottom border", min_value=0, max_value=size_y, value=size_y)
        bounding_box = [left_border, top_border, right_border, bottom_border]
    else:
        bounding_box = None
        


col1, col2 = st.columns(2)
with col1:
    uploaded_calib_files = st.file_uploader(
        "Upload Calibration Images", type=["png"], accept_multiple_files=True)
with col2:
    uploaded_data_files = st.file_uploader(
        "Upload Bias Images", type=["png"], accept_multiple_files=True)
        
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
                img_array, f"{uploaded_calib_file.name}", color_map, color_range, 
                fig_height=fig_height, bounding_box=bounding_box
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
    # st.write(calib_img_arrays.keys())
    calib_parallel_on = calib_img_arrays["calib_parallel_on"]
    calib_parallel_off = calib_img_arrays["calib_parallel_off"]
    calib_cross_on = calib_img_arrays["calib_cross_on"]
    calib_parallel_minus_cross = np.subtract(calib_img_arrays["calib_parallel_on"], calib_img_arrays["calib_cross_on"])
    with st.expander("Parallel minus Cross"):
        vmin, vmax = np.percentile(calib_parallel_minus_cross, q=global_range_pctl)
        color_range = st.slider("Color range", min_value=0.0, max_value=65500.0, value=[vmin, vmax])
        st.plotly_chart(heatmap_plot_with_bounding_box(
            calib_parallel_minus_cross, "Parallel minus Cross", color_map, color_range, 
            fig_height=fig_height, bounding_box=bounding_box))


#######################
### DATA PROCESSING ###
#######################
raw_image_plotly_figures = {}
E_field_arrays = {}
E_field_plotly_figures = {}

# save_dir = r"DATA/Saved_Efield"

with st.expander("Control Panel"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Emin = st.number_input("Minimum E-field (MV/m)", min_value=0.0, max_value=10.0, value=0.0)*1e6
        Emax = st.number_input("Maximum E-field (MV/m)", min_value=0.0, max_value=10.0, value=1.0)*1e6
    with col2:
        show_raw_image = st.checkbox("Show raw image", value=True, key=f"show_raw_image")
        perform_T_normalization = st.checkbox("Normalize transmission (0<T<1)", value=True, key=f"normalize_transmission")
        show_transmission_image = st.checkbox("Show transmission image", value=False, key=f"show_transmission_image")
        calculate_E_field = st.checkbox("Calculate E-field", value=True, key=f"do_E_field")
    # with col3:
    #     normalization_method = st.radio("Normalization method", 
    #                                     options=["I_bias/I_para", "(I_bias-I_cross)/(I_para)", "(I_bias-I_cross)/(I_para-I_off)"])
    #     save_dir = st.text_input("Save directory", value=save_dir)
    # with col4:

if uploaded_data_files:
    for uploaded_data_file in uploaded_data_files:
        filename = remove_extension(uploaded_data_file.name)
        img_array = png_to_array(uploaded_data_file)
        img_array = crop_image(img_array, crop_range_x, crop_range_y)

        if calib_img_arrays:
            numerator = (img_array 
            - calib_img_arrays["calib_cross_on"])
            denominator = (
                calib_img_arrays["calib_parallel_on"] 
                - calib_img_arrays["calib_parallel_off"]
            )
            denominator = remove_low_value_pixels(denominator)
            T_array = numerator / denominator

        with st.expander(f"Plot {filename}"):
            
            vmin, vmax = np.percentile(img_array, q=global_range_pctl)
            color_range = st.slider("Color range", min_value=0.0, max_value=65500.0, value=[vmin, vmax],
                                    key=f"raw_image_color_range_{filename}")
            img_fig = heatmap_plot_with_bounding_box(
                img_array, f"{uploaded_data_file.name}", color_map, color_range, 
                fig_height=fig_height, bounding_box=bounding_box
            )
   
            raw_image_plotly_figures[filename] = img_fig
            if show_raw_image:
                st.plotly_chart(img_fig)
            if calib_img_arrays:

                if perform_T_normalization:
                    T_array = cap_array(T_array, 0, 1.0)
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
                        bounding_box=bounding_box,
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
                        bounding_box=bounding_box,
                    )
                    E_fig.update_layout(
                        coloraxis_colorbar=dict(
                            title="E-field (V/m)",
                            tickvals=[Emin, Emax],
                            ticktext=[f"{Emin:.2e}", f"{Emax:.2e}"]
                        )
                    )
                    E_field_plotly_figures[f"{filename}"] = E_fig # save the plotly figure
                    st.plotly_chart(E_fig)

# st.write(E_field_arrays.keys())

if st.button("Save Raw Images as Plotly Figures"):
    for filename, img_fig in raw_image_plotly_figures.items():
        save_plotly_figure(img_fig, f"{filename}_raw", save_dir=r"DATA/Saved_Images")

save_dir = st.text_input("Save directory", value=r"DATA/Saved_Efield")
if st.button("Save E-field data"):
    for filename, E_field_array in E_field_arrays.items():
        save_array_to_png(E_field_array, f"E-field_{filename}", save_dir=save_dir)
        save_plotly_figure(E_field_plotly_figures[filename], f"E-field_{filename}", save_dir=save_dir)

# with st.expander("All E-field plots"):
#     if E_field_arrays:
#         # Calculate grid dimensions
#         n_plots = len(E_field_arrays)
#         n_cols = 1 
#         n_rows = n_plots
        
#         # Create subplot figure
#         fig = make_subplots(rows=n_rows, cols=n_cols, 
#                            subplot_titles=list(E_field_plotly_figures.keys()),
#                            start_cell="top-left",
#                            horizontal_spacing=0.05,
#                            vertical_spacing=0.05)  # Equal height for all rows
        
#         # Add each E-field plot as a subplot
#         for i, (filename, E_field_array) in enumerate(E_field_arrays.items()):
#             row = i + 1
#             col = 1

#             heatmap = create_plotly_figure(E_field_array, 
#                                            title=f"E-field_{filename}", 
#                                            cmap=color_map, 
#                                            color_range=[Emin, Emax])
#             fig.add_trace(heatmap, row=row, col=col)
            
#             # Update subplot layout
#             # fig.update_xaxes(title_text="x", row=row, col=col)
#             # fig.update_yaxes(title_text="y", row=row, col=col)
        
#         # Update overall layout
#         fig.update_layout(
#             height=(fig_height) * n_rows,  # Total height based on number of rows
#             # width=fig_height,  # Fixed width
#             title_text="E-field Maps",
#             margin=dict(l=50, r=50, t=100, b=50),  # Add margins
#             # coloraxis=dict(
#             #     colorbar=dict(
#             #         title="E-field (V/m)",
#             #         tickvals=[Emin, Emax],
#             #         ticktext=[f"{Emin:.2e}", f"{Emax:.2e}"]
#             #     )
#             # )
#         )
        
#         st.plotly_chart(fig)
#     else:
#         st.write("No E-field plots available")
