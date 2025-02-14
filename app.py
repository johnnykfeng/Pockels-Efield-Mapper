import streamlit as st


home_page = st.Page("main.py", title="Pockels Image Analyzer", icon="🔍")
processed_efields_page = st.Page("processed_efields.py", title="Processed E-fields", icon="📈")

pg = st.navigation([home_page, processed_efields_page])
pg.run()
