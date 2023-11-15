import streamlit as st
from streamlit_gallery import apps, components
from streamlit_gallery.pages.page import page_group
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

st.set_page_config(page_title="FasterX", page_icon="🎈", layout="wide")

def main():
    page = page_group("p")
    
    with st.sidebar:
        st.title("FasterX")

        with st.expander("🏠 HOME", True):
            page.item("Home", apps.home, default=True)

        with st.expander("🧩 FEATURES", True):
            page.item("Stock Detection", components.stock)
            page.item("Production Sequence", components.sequence_production)
            page.item("About", components.about)
                

    page.show()

if __name__ == "__main__":
    main()
