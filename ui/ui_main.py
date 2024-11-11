import streamlit as st
from streamlit_file_browser import st_file_browser
from streamlit_pdf_viewer import pdf_viewer
from streamlit_dimensions import st_dimensions

from pathlib import Path
import re

st.set_page_config(layout="wide")


st.markdown("""
<style>

[data-testid="stMarkdown"] h1 {
    font-size:2rem !important;
}
</style>
""", unsafe_allow_html=True)

ROOT = "./references"
SCROLL_HEIGHT = 1000
# Create a file browser component in the sidebar
with st.sidebar:
    st.markdown("## File Browser")
    event = st_file_browser(
        path=ROOT,
        key="file_browser",
        extentions=["md"],
        show_choose_file=False,
        show_download_file=False,
        show_preview=False,
        use_cache=True,
    )

pdf_content = None
content = None

# Main content area
if event and event['type'] == 'SELECT_FILE':
    file_path = Path(ROOT, event['target']['path'])
    file_name = file_path.stem
    pdf_path = file_path.parent / (file_name.replace('_overview', '') + '.pdf')
    
    with open(file_path, 'r') as file:
        content = file.read()
        content = re.sub(r'\*\*(.*?)\*\*', r':orange[\1]', content)
        
    with open(pdf_path, 'rb') as file:
        pdf_content = file.read()

col1, col2 = st.columns([1.5,1])
            

if content and pdf_content:
    with col1.container(height=SCROLL_HEIGHT):
        
        pdf_viewer(pdf_content, width=st_dimensions("col1")['width'], 
                   height=SCROLL_HEIGHT-100)
    with col2.container(height=SCROLL_HEIGHT):
        st.markdown(content)

    