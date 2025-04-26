import streamlit as st

image_cropper_page = st.Page("image_cropper.py", title="Image Cropper", icon="🔍")
home_page = st.Page("main.py", title="Pockels Image Analyzer", icon="🔍")
processed_efields_page = st.Page("processed_efields.py", title="Processed E-fields", icon="📈")

pg = st.navigation([image_cropper_page, home_page, processed_efields_page])
pg.run()
