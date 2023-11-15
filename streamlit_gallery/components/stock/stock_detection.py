# import all the tools we need
import urllib
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import os 
from PIL import Image
import random
import xml.etree.ElementTree as ET
import time
import requests
import streamlit as st
from streamlit_folium import folium_static
import folium
import app_component as ac 
import faster_detection as fd
#import app_user as uv
import time




def main():
    ac.render_cta()

    # List of image URLs and their corresponding content
    image_data = [
        {
            "url": "images/idea.jpg",
            "content": "The model will read objects in an image using bounding boxes. Green bounding boxes will provide information about stock availability, while red bounding boxes will provide information about empty stock. Data from the empty stock detection results will be used to fill the pallet packaging."
        },
        {
            "url": "images/compare r-cnn.jpg",
             "content": """
            **R-CNN (Region-based Convolutional Neural Network):**
            R-CNN is an early object detection model that involves dividing an image into regions, extracting features from each region, and then classifying and refining bounding boxes. It uses an external algorithm, like selective search, for region proposal generation.

            **Fast R-CNN:**
            Fast R-CNN improves on R-CNN by introducing Region of Interest (ROI) pooling. It shares convolutional features across all region proposals, making the process more efficient. The image is processed only once, and region proposals are obtained from the feature map.

            **Faster R-CNN:**
            Faster R-CNN integrates the Region Proposal Network (RPN) directly into the model. RPN predicts region proposals alongside object classification and bounding box regression in a single architecture, making it faster and end-to-end trainable for object detection.
        """
        },
        {
            "url": "images/iou.jpg",
             "content": """
            **Iou**
            IoU is a metric used to measure the extent to which two areas overlap. In the context of object detection, IoU is often used to evaluate how much the bounding box of the model's prediction overlaps with the actual ground truth bounding box.

            Intersection Area: The area where the model's prediction and the ground truth overlap.
            Union Area: The total area covered by both the model's prediction and the ground truth.
            IoU values range from 0 to 1, where 0 indicates no overlap, and 1 indicates perfect overlap.

            In this project, an IoU threshold value of 0.8 indicates that the model will only consider its prediction correct if more than 80% of the prediction area overlaps with the ground truth area.

            The dataset for this project consists of 360 images in the training set and 40 images in the evaluation set. To measure the performance of the object detection model, the Mean Average Precision (MAP) metric is employed. The trained model exhibits good performance, with a MAP value of 0.88 for the training data and 0.90 for the evaluation data. Mean Average Precision reflects the model's ability to predict objects by considering both crucial aspects: accuracy (precision) and recall (sensitivity). With these high MAP values, it can be inferred that the model is capable of providing accurate predictions and effectively identifying objects in both the training and evaluation datasets
            """
        }
        
        ]
    
    st.title("Stock Detection")

    from_tab1, from_tab2 = st.tabs(
        ["Project Idea", "Object Detection"]
    )
    

    with from_tab1:
        #st.markdown("<style>div { text-align: center; }</style>", unsafe_allow_html=True)
        st.markdown("#### Project Idea")
        selected_image_index = 0

        # Display the selected image
        st.image(image_data[selected_image_index]["url"], use_column_width=True)

        # Display the corresponding content
        st.write(image_data[selected_image_index]["content"])

        st.markdown("#### Comparison of Object Detection Model Architectures")
        selected_image_index = 1

        # Display the selected image
        st.image(image_data[selected_image_index]["url"], use_column_width=True)

        # Display the corresponding content
        st.write(image_data[selected_image_index]["content"])

        st.markdown("#### Evaluation Metric")
        selected_image_index = 2

        # Display the selected image
        st.image(image_data[selected_image_index]["url"], use_column_width=True)

        # Display the corresponding content
        st.write(image_data[selected_image_index]["content"])

    with from_tab2:
        #st.title("Stock Detection")
        # Tambahkan tautan unduh untuk sampel gambar
        st.markdown("[Download Sample Piture](https://drive.google.com/u/0/uc?id=1dNpr6S2Jv66GjdZlaXA-2lqtz9FmNZwj&export=download)")

        fstrx_detection = fd.FasterDetection()
        image = fstrx_detection.load_image()
        result = st.button('Run on image')
        if result:
            with st.spinner('Processing...'):
                if isinstance(image, Image.Image):  # Memeriksa tipe data
                    # Prediction
                    test_img, test_boxes, test_labels = fstrx_detection.single_img_predict(image)

                    # Memanggil fungsi draw_boxes
                    test_img, test_data = fstrx_detection.draw_boxes(test_img, test_boxes, test_labels)

                    test_output = np.clip(test_img, 0.0, 1.0)


                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Display the result
                        st.image(test_output, caption='Prediction', use_column_width=True)

                    with col2:
                        # Menampilkan data
                        st.write('Predicted Labels:')
                        stock_data = [part_info for part_info in test_data if part_info['Part Name'] == 'cowltop']

                        if 1 in test_labels:
                            with st.expander("Predicted Stock Info:", expanded=False):
                                for part_info in stock_data:
                                    st.write(part_info)

                        if 2 in test_labels:
                            empty_pallet_data = [part_info for part_info in test_data if part_info['Part Name'] == 'empty_pallet']
                            with st.expander("Predicted Empty_Pallet:", expanded=False):
                                for part_info in empty_pallet_data:
                                    st.write(part_info)
                else:
                    st.write('Invalid image format. Please upload a valid image.')
        
        # Contoh bounding box yang diprediksi dan sebenarnya
        
        
        #true_box = [15, 15, 30, 30]

        # Hitung IoU
        #iou = calculate_iou(predicted_box, true_box)

if __name__ == "__main__":
    main()

