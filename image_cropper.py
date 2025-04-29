import streamlit as st
import st_tailwind as tw
import numpy as np
import os
from PIL import Image
from image_process import png_to_array, crop_image, save_array_to_png
from plotting_modules import create_plotly_figure, colored_pockels_images_matplotlib
import time
import matplotlib.pyplot as plt
from matplotlib import patches
import zipfile
import io
st.set_page_config(layout="wide")
tw.initialize_tailwind()

st.title("Image Cropper")
with tw.container(classes="bg-green-100 text-lg font-bold text-blue-900 rounded-lg"):
    with st.expander("Image Processing"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            crop_range_left = st.number_input("Crop range left", min_value=0, max_value=639, value=160)
        with col2:
            crop_range_right = st.number_input("Crop range right", min_value=0, max_value=639, value=510)
        with col3:
            crop_range_top = st.number_input("Crop range top", min_value=0, max_value=511, value=190)
        with col4:
            crop_range_bottom = st.number_input("Crop range bottom", min_value=0, max_value=511, value=310)
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

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
    
if "img_arrays" not in st.session_state:
    st.session_state.img_arrays = {}
    
if "mat_fig" not in st.session_state:
    st.session_state.mat_fig = None

with st.sidebar:
    uploaded_png_files = st.file_uploader("Upload PNG files", type=["png"], 
                                      accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")

col1, col2 = st.columns(2)
with col1:
    if st.button("Clear Uploaded Files"):
        st.session_state.uploader_key += 1
        time.sleep(0.5)
        st.rerun()
    
with col2:
    save_container = st.empty()
    download_btn = st.empty()

# Crop images and save cropped images to session state
if uploaded_png_files:
    st.session_state.img_arrays = {}
    for uploaded_png_file in uploaded_png_files:
        img_array = png_to_array(uploaded_png_file)
        img_array = crop_image(img_array, crop_range_x, crop_range_y)
        st.session_state.img_arrays[uploaded_png_file.name] = img_array
        
with tw.container(classes="bg-gray-100 rounded-lg"):
    with st.expander("Cropped images"):
        for uploaded_png_file in uploaded_png_files:
            img_array = st.session_state.img_arrays[uploaded_png_file.name]
            vmin, vmax = np.percentile(img_array, (5, 99))
            color_range = st.slider("Color range", min_value=0.0, max_value=vmax*1.5, value=[vmin, vmax])
            fig = create_plotly_figure(img_array, title=uploaded_png_file.name, color_range=color_range)
            if apply_bounding_box:
                fig.add_shape(type="rect",
                            x0=bounding_box[0], y0=bounding_box[1], x1=bounding_box[2], y1=bounding_box[3],
                            line=dict(color="white", width=2, dash="dash"))
            st.plotly_chart(fig, title=uploaded_png_file.name)


if uploaded_png_files:
    
    with st.expander("Matplotlib plots"):
        vmin, vmax = np.percentile(st.session_state.img_arrays["calib_parallel_on.png"], (10, 90))
        col1, col2 = st.columns(2)
        with col1:
            color_range_radio = st.radio("Color range", options=["Auto", "Fixed"], index=0, key="color_range_radio")
        with col2:
            color_min, color_max = st.slider("Color range", min_value=0.0, max_value=vmax*1.5, value=[vmin, vmax], key="color_range_matplotlib")

        mat_fig = colored_pockels_images_matplotlib(st.session_state.img_arrays, 
                                          color_range_radio, 
                                          color_min, color_max, 
                                          apply_bounding_box, bounding_box)
        st.session_state.mat_fig = mat_fig

        st.pyplot(fig = mat_fig)

with save_container.popover("SAVE FILES"):
    # save_dir = st.text_input("Save directory")
    # if save_dir:
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #         st.success(f"Created directory: {save_dir}")
    #     if uploaded_png_files:
    #         if st.button("Save cropped images") and uploaded_png_files:
    #             for uploaded_png_file in uploaded_png_files:
    #                 img_array = st.session_state.img_arrays[uploaded_png_file.name]
    #                 save_array_to_png(img_array, f"{save_dir}/{uploaded_png_file.name}")
    #             st.success("Saved cropped images!")
    #         if st.button("Save matplotlib figure"):
    #             mat_fig.savefig(f"{save_dir}/Matplotlib_figure_{color_range_radio}-color.pdf", format="pdf")
    #             st.success("Saved matplotlib figure!")
    #     else:
    #         st.warning("No files uploaded")

    if uploaded_png_files:
        # Create a zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for uploaded_png_file in uploaded_png_files:
                img_array = st.session_state.img_arrays[uploaded_png_file.name]
                # Create a temporary file in memory
                img_buffer = io.BytesIO()
                img = Image.fromarray(img_array)
                img.save(img_buffer, format='PNG')
                # Add the image to the zip file
                zip_file.writestr(uploaded_png_file.name, img_buffer.getvalue())
        
        # Prepare the zip file for download
        zip_buffer.seek(0)
        st.download_button(
            label="Click to download ZIP of cropped images",
            data=zip_buffer,
            file_name="cropped_images.zip",
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
                file_name=f"Matplotlib_figure_{color_range_radio}-color.pdf",
                mime="application/pdf"
            )



