import streamlit as st 
import app_component as ac



def main():

    ac.render_cta()

    # copies 
    home_title = "Faster"

    home_introduction = "Welcome to FasterX. The application we have developed has a specific purpose in supporting stock detection within the production sequence. We employ the Faster R-CNN method with the ResNet-50 neural network architecture from PyTorch as the foundation to achieve this goal. With this approach, the application can efficiently and accurately detect stock items associated with the production sequence, enabling companies to better manage their inventory, enhance production efficiency, and reduce the risk of inventory shortages. This solution can provide significant added value in complex production environments."
    home_privacy = "This website, the data and the gentrification model are the result of the capstone project of DCD, vulcan Class of 2023 at Algoritma Data Science School."

    st.markdown(
        "<style>#MainMenu{visibility:hidden;}</style>",
        unsafe_allow_html=True
    )

    #st.title(home_title)
    motto_size = 30  # Sesuaikan ukuran motto sesuai keinginan Anda
    margin_top = 1  # Sesuaikan jarak atas sesuai keinginan Anda (dalam piksel)

    st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=20>X</font></span>\n\n<small><span style="font-size:{motto_size}px; margin-top:{margin_top}px;">Where Speed Meets Precision</span></small>""", unsafe_allow_html=True)


    st.markdown("""\n""")
    st.markdown("#### Greetings")
    st.write(home_introduction)


    # List of image URLs and their corresponding content
    image_data = [
        {
            "url": "images/Slide1.jpg",
            "content": "This project delves into how our 'Stock Availability Detection in the Supply Sequence System with Faster R-CNN' project adapts to the challenges presented by the introduction of Mitsubishi Motor's latest product, the X-Force model. The primary goal is to optimize the supply chain, ensuring it can effectively respond to the ever-evolving demands of the market in light of this new product launch. This initiative underscores our commitment to staying agile and responsive in an ever-changing business landscape."
        },
        {   "url": "images/image2.jpg",
            "content": "1. **Stamping Stage:**\n   - Metal sheets are shaped with precision using advanced techniques.\n"
                "2. **Welding Stage:**\n   - Components are meticulously integrated through welding processes, forming a strong and robust structure.\n"
                "3. **Painting Stage:**\n   - The Painting Stage involves applying a high-quality paint finish to the vehicle's body, ensuring both aesthetic appeal and protection against external elements.\n"
                "4. **Assembly Stage:**\n"
                "   - Formed components are seamlessly brought together to create a complete vehicle.\n"
                "   - Mechanical, electrical, and interior systems are meticulously installed with attention to detail.\n"
                "5. **Inspection Stage:**\n   - Every aspect of the vehicle is thoroughly examined to meet established safety and performance standards.\n"
                "6. **Shipping Stage:**\n   - The vehicle is carefully packaged and delivered to customers, aiming to provide an exceptional driving experience."
        },
        {
            "url": "images/tf1.gif",
            "content": "Phantom inventory refers to a situation where a company's inventory system reports the presence of certain goods, but these goods are not actually physically on hand or available for sale. This often occurs due to discrepancies between inventory records in the computer system and the reality in the physical warehouse or store. The main causes of phantom inventory can include counting errors, differences between data recorded in the system and missing or damaged items, or issues with inventory management processes. Phantom inventory can be a serious problem as it can lead to incorrect inventory management decisions, unnecessary expenses, and customer dissatisfaction when expected items are not available."
        },
        {
            "url": "images/tf2.gif",
            "content": "In a similar vein to traffic jam when the number of vehicles is not balanced with road expansion, it can cause detrimental traffic jam. Likewise, in the management of automotive component stock, an imbalance can lead to similar problems. Imbalances in automotive component inventory can have serious consequences. Excess stock may be forgotten or damaged, while the required components may not be available in sufficient quantities. As a result, downtime or production line stops can occur, ultimately leading to the loss of customers and a decline in company revenue."
        },
        {
            "url": "images/img3.jpg",
            "content": "With the launch of the new model, Mitsubishi X-Force, the stock flow will increase, while the storage capacity remains unchanged. This situation has the potential to create a number of issues, one of which is the possibility of excess stock that can lead to storage confusion. However, with a change in the supply sequence system based on production order, the potential to optimize the smooth and efficient flow of stock will be open."
        },
        {
            "url": "images/img4.jpg",
            "content": "Just like traffic police officers regulate vehicle flow, here I aim to create a model that can detect objects using the Faster R-CNN method. This model will control the stock flow, enabling the company to automatically monitor and manage inventory. With a more efficient approach, it can minimize the risk of excess stock or stock shortages, improve storage efficiency, and optimize the overall business process."
        }
    ]

    from_tab1, from_tab2, from_tab3, from_tab4 = st.tabs(
        ["Cover", "Business Process", "Problem Statement", "Project Idea"]
    )

    with from_tab1:

        # Create a slider to navigate through images
        #selected_image_index = st.slider("Background Problem", 0, len(image_data) - 1, 0)
        selected_image_index = 0

        # Display the selected image
        st.image(image_data[selected_image_index]["url"], use_column_width=True)

        # Display the corresponding content
        st.write(image_data[selected_image_index]["content"])

    with from_tab2:
        # Create a slider to navigate through images
        #selected_image_index = st.slider("Background Problem", 0, len(image_data) - 1, 0)
        selected_image_index = 1

        # Display the selected image
        st.image(image_data[selected_image_index]["url"], use_column_width=True)

        # Display the corresponding content
        st.write(image_data[selected_image_index]["content"])
    
    with from_tab3:

        st.markdown("#### 1. Phantom Inventory")
        # Create a slider to navigate through images
        #selected_image_index = st.slider("Background Problem", 0, len(image_data) - 1, 0)
        selected_image_index = 2

        # Display the selected image
        st.image(image_data[selected_image_index]["url"], use_column_width=True)

        # Display the corresponding content
        st.write(image_data[selected_image_index]["content"])


        st.markdown("#### 2. The Impact of Stock Imbalance")
        selected_image_index = 3

        # Display the selected image
        st.image(image_data[selected_image_index]["url"], use_column_width=True)

        # Display the corresponding content
        st.write(image_data[selected_image_index]["content"])

        st.markdown("#### 3. Increase in Stock Level")
        selected_image_index = 4

        # Display the selected image
        st.image(image_data[selected_image_index]["url"], use_column_width=True)

        # Display the corresponding content
        st.write(image_data[selected_image_index]["content"])

    with from_tab4:
        
        # Create a slider to navigate through images
        #selected_image_index = st.slider("Background Problem", 0, len(image_data) - 1, 0)
        selected_image_index = 5
        col1, col2 = st.columns([1, 3])

        with col1:
            # Display the selected image
            st.image(image_data[selected_image_index]["url"], use_column_width=True)

        with col2:
            # Display the corresponding content
            st.write(image_data[selected_image_index]["content"])

    st.markdown("#### Privacy")
    st.write(home_privacy)

if __name__ == "__main__":
    main()