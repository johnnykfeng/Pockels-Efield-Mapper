import streamlit as st

efield_compute_page = st.Page("compute_efield_page.py", title="E-field Compute", icon="🧮")
image_cropper_page = st.Page("process_image_page.py", title="Image Cropper", icon="🎦")
# efield_compute_10mm_page = st.Page("Efield_compute_10mm.py", title="E-field Compute 10mm", icon="🧮")
processed_efields_page = st.Page("processed_efields_page.py", title="Processed E-fields", icon="📈")

pg = st.navigation([
    efield_compute_page, 
    image_cropper_page, 
    processed_efields_page
    ])
pg.run()
