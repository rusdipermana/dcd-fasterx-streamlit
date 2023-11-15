import streamlit as st
import app_component as ac


#import streamlit as st

def main():
    # Judul halaman
    st.title("About")

    from_tab1, from_tab2 = st.tabs(
        ["Summary", "About Me"]
    )
    with from_tab1:
        st.markdown("""**Insight:** The integration of Faster R-CNN into MMKI’s supply sequence system holds the potential for a significant positive impact on the business operations of MMKI. Through the effective identification and resolution of empty stock situations using advanced visual recognition, MMKI anticipates a substantial decrease in production disruptions and delays. This heightened accuracy in stock monitoring, made possible by Faster R-CNN, is poised to optimize inventory management and streamline the entirety of the production process. As a result, MMKI stands to achieve smoother operations, minimized downtime, and enhanced resource allocation, leading to increased operational efficiency and improved overall performance.
    """)

        st.markdown("""**Recommendation:** In an effort to optimize stock control, it is crucial for companies to seek effective solutions. These solutions should be able to monitor and manage inventory accurately, prevent stockouts or overstocking, and enhance operational efficiency. With technological advancements, the use of tools like object detection and advanced data analytics has become promising options. With this approach, companies can ensure their inventory is always under control, avoiding potential disruptions in production, and improving their overall performance. Therefore, companies in various sectors should continually search for solutions that suit their needs to achieve effective stock control.
    """)
    
    with from_tab2:
        col1, col2 = st.columns([1, 3])

        with col1:
            # Container untuk gambar profil dan teks
            st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; flex-direction: column; text-align: center;">
                <img src="https://drive.google.com/uc?id=1EUbPKr0w3wWNwxogWJ6Ynv2y_S70rk_E" alt="Rusdi Permana" style="width: 50%; height: 50%;">
                <h3>Rusdi Permana</h3>
                <p>Supply Chain Specialist</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Deskripsi atau teks tentang halaman "About"
            st.markdown("""
            Rusdi Permana graduated from Nusa Mandiri with a degree in Information Systems and has accumulated 13 years of work experience in the automotive industry. I firmly believe that the accessibility of data science can have a positive impact on productivity within the industrial sector. With my experience in leading teams in the workplace and using skills in Python and R programming as primary tools, I am convinced that gathering relevant information, identifying problems, and conducting thorough and composed problem analyses are the keys to solving various challenges.
            """)
    
    ac.render_cta()

if __name__ == "__main__":
    main()

