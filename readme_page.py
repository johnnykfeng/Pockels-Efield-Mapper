import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

pdf_path = "assets/Pockels_Efield_Mapper_readme.pdf"
with st.expander("How Pockels Efield Mapper works", expanded=False):
    # st.write("hello")
    with open(pdf_path, "rb") as pdf_file:
        pdf_viewer(pdf_file.read(), annotations=[])


# Read README.md content
with open("README.md", "r", encoding="utf-8") as f:
    readme_content = f.read()
# Display README content as markdown
st.markdown(readme_content)
