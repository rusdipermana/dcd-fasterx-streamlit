import streamlit as st
import streamlit.components.v1 as c 
import random as r 


def render_cta():
  with st.sidebar:
        st.write("Let's connect!")
        linkedin_url = "http://linkedin.com/in/rusdipermana/"
        rpubs_url = "https://www.rpubs.com/rusdipermana"
        github_url = "https://github.com/rusdipermana"
        
        # LinkedIn button
        st.markdown(f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_url})", unsafe_allow_html=True)

        # RPubs button with custom FontAwesome icon
        st.markdown(f"[![RPubs](https://img.shields.io/badge/RPubs-151515?style=for-the-badge&logo=r&logoColor=white)]({rpubs_url})", unsafe_allow_html=True)
        
         # GitHub button
        st.markdown(f"[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)]({github_url})", unsafe_allow_html=True)
      
