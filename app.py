import streamlit as st

image_cropper_page = st.Page("image_cropper.py", title="Image Cropper", icon="🎦")
efield_compute_page = st.Page("Efield_compute.py", title="E-field Compute", icon="🧮")
processed_efields_page = st.Page("processed_efields.py", title="Processed E-fields", icon="📈")

pg = st.navigation([image_cropper_page, efield_compute_page, processed_efields_page])
pg.run()
