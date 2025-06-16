import streamlit as st
import numpy as np
import os
from PIL import Image
from modules.image_process import png_to_array, crop_image
from modules.plotting_modules import create_plotly_figure, colored_pockels_images_matplotlib
import time
import zipfile
import io
st.set_page_config(layout="wide")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
    
if "img_arrays" not in st.session_state:
    st.session_state.img_arrays = {}
    
if "mat_fig" not in st.session_state:
    st.session_state.mat_fig = None

@st.cache_data
def cached_colored_pockels_images_matplotlib(images_dict: dict, 
                                              color_range_radio: str, 
                                              color_min: float, 
                                              color_max: float, 
                                              apply_bounding_box: bool, 
                                              bounding_box: tuple):
    return colored_pockels_images_matplotlib(images_dict, color_range_radio, color_min, color_max, apply_bounding_box, bounding_box)
    
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
        left_border = st.number_input("Left border", min_value=0, max_value=size_x, value=32)
    with col2:
        right_border = st.number_input("Right border", min_value=0, max_value=size_x, value=373)
    with col3:
        top_border = st.number_input("Top border", min_value=0, max_value=size_y, value=32)
    with col4:
        bottom_border = st.number_input("Bottom border", min_value=0, max_value=size_y, value=98)
    bounding_box = [left_border, top_border, right_border, bottom_border]
        
    apply_bounding_box = st.checkbox("Apply bounding box", value=True)


with st.sidebar:
    upload_choice = st.radio("Upload choice", options=["Upload PNG files", "Choose from folder"], index=0)
    if upload_choice == "Upload PNG files":
        uploaded_png_files = st.file_uploader("Upload PNG files", type=["png"], 
                                      accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
    elif upload_choice == "Choose from folder":
        raw_data_folder = st.text_input("Enter folder path", value="data")
        uploaded_png_files = [os.path.join(raw_data_folder, f) for f in os.listdir(raw_data_folder) if f.endswith('.png')]

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Clear Uploaded Files"):
        st.session_state.uploader_key += 1
        time.sleep(0.5)
        st.rerun()
with col2:
    sensor_id = st.text_input("Sensor ID", value="sensor-id")

with col3:
    save_container = st.empty()
    download_btn = st.empty()

# Crop images and save cropped images to session state
if uploaded_png_files:
    st.session_state.img_arrays = {}
    for uploaded_png_file in uploaded_png_files:
        img_array = png_to_array(uploaded_png_file)
        img_array = crop_image(img_array, crop_range_x, crop_range_y)
        if upload_choice == "Upload PNG files":
            filename = uploaded_png_file.name
        elif upload_choice == "Choose from folder":
            filename = os.path.basename(uploaded_png_file)
        st.session_state.img_arrays[filename] = img_array
        
with st.expander("Cropped images"):
    for img_array, filename in zip(st.session_state.img_arrays.values(), st.session_state.img_arrays.keys()):

        vmin, vmax = np.percentile(img_array, (5, 99))
        color_range = st.slider("Color range", min_value=0.0, max_value=vmax*1.5, value=[vmin, vmax], key=f"color_range_{filename}")
        fig = create_plotly_figure(img_array, title=filename, color_range=color_range)
        if apply_bounding_box:
            fig.add_shape(type="rect",
                        x0=bounding_box[0], y0=bounding_box[1], x1=bounding_box[2], y1=bounding_box[3],
                        line=dict(color="white", width=2, dash="dash"))
        st.plotly_chart(fig, title=filename)


if st.session_state.img_arrays:
    
    with st.expander("Matplotlib plots"):
        vmin, vmax = np.percentile(st.session_state.img_arrays["calib_parallel_on.png"], (10, 90))
        col1, col2 = st.columns(2)
        with col1:
            color_range_radio = st.radio("Color range", options=["Auto", "Fixed"], index=0, key="color_range_radio")
        with col2:
            color_min, color_max = st.slider("Color range", min_value=0.0, max_value=vmax*1.5, value=[vmin, vmax*0.8], key="color_range_matplotlib")

        mat_fig = cached_colored_pockels_images_matplotlib(st.session_state.img_arrays, 
                                          color_range_radio, 
                                          color_min, color_max, 
                                          apply_bounding_box, bounding_box)
        mat_fig.suptitle(f"Raw Camera Images with {color_range_radio}-scale color range", fontsize=15, y=1.05)
        st.session_state.mat_fig = mat_fig
        st.pyplot(fig = mat_fig)

with save_container.popover("SAVE FILES"):
    
    if st.session_state.img_arrays:
        # Create a zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for img_array, filename in zip(st.session_state.img_arrays.values(), st.session_state.img_arrays.keys()):
                # Create a temporary file in memory
                img_buffer = io.BytesIO()
                # Convert array to uint16 (16-bit unsigned integer) which is compatible with PNG
                img_array = img_array.astype(np.uint16)
                img = Image.fromarray(img_array)
                img.save(img_buffer, format='PNG')
                # Add the image to the zip file
                zip_file.writestr(filename, img_buffer.getvalue())
        
        # Prepare the zip file for download
        zip_buffer.seek(0)
        st.download_button(
            label="Click to download ZIP of cropped images",
            data=zip_buffer,
            file_name=f"{sensor_id}_cropped_images.zip" \
                if sensor_id else "cropped_images.zip",
            mime="application/zip"
        )

        # Only try to save the matplotlib figure if it exists
        if st.session_state.mat_fig is not None:
            # Create a PDF buffer for the matplotlib figure
            pdf_buffer = io.BytesIO()
            st.session_state.mat_fig.savefig(pdf_buffer, format='pdf', bbox_inches='tight')
            pdf_buffer.seek(0)

            # Add download button for PDF figure
            st.download_button(
                label="Download Matplotlib figure as PDF",
                data=pdf_buffer,
                file_name=f"{sensor_id}_Raw_Camera_Images_{color_range_radio}-color.pdf" \
                    if sensor_id else f"Raw_Camera_Images_{color_range_radio}-color.pdf",
                mime="application/pdf"
            )



