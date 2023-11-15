import gdown
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import streamlit as st
import os
import numpy as np
import xml.etree.ElementTree as ET


class FasterDetection():
    def __init__(self) -> None:
        self.dataset = None
        num_classes = 3
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        pre_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = pre_model.roi_heads.box_predictor.cls_score.in_features
        pre_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        pre_model = pre_model.to(device)

        model_filename = 'model_30.pth'

        if not os.path.isfile(model_filename):
            # Ganti tautan Google Drive dengan tautan tautan berbagi yang sesuai
            gdrive_url = 'https://drive.google.com/u/0/uc?id=1GLKN0xaNOZvv6fR1sTna091MgiVf3f0I&export=download'

            # Unduh model dari Google Drive
            gdown.download(gdrive_url, model_filename, quiet=False)

        # Muat model state_dict dari model yang diunduh
        state_dict = torch.load(model_filename, map_location=device)
        pre_model.load_state_dict(state_dict)

        self.frcnn_model = pre_model

    def load_image(self):     
        uploaded_file = st.file_uploader(label='Pick an effluent dialysate image', type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption='Uploaded Image', use_column_width=True)
            return img

    def single_img_predict(self, img, nm_thrs=0.3, score_thrs=0.3):
        test_img = transforms.ToTensor()(img)
        self.frcnn_model.eval()

        with torch.no_grad():
            device = next(self.frcnn_model.parameters()).device
            predictions = self.frcnn_model([test_img.to(device)])

        test_img = test_img.permute(1, 2, 0).numpy()

        keep_boxes = torchvision.ops.nms(predictions[0]['boxes'], predictions[0]['scores'], nm_thrs)
        score_filter = predictions[0]['scores'][keep_boxes] > score_thrs
        test_boxes = predictions[0]['boxes'][keep_boxes][score_filter].cpu().numpy()
        test_labels = predictions[0]['labels'][keep_boxes][score_filter].cpu().numpy()

        return test_img, test_boxes, test_labels

    def draw_boxes(self, img, boxes, labels, thickness=3):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Konversi dari RGB ke BGR

        data = []  # List untuk mengumpulkan data

        for nomor_box, (box, label) in enumerate(zip(boxes, labels), start=1):
            box = [int(x) for x in box]
            if label == 1:
                color = (0, 255, 0)  # Hijau untuk "cowltop"
                part_info = {'No. ': nomor_box, 'Part Name': 'cowltop', 'Stock Info': '238 Pcs'}
            elif label == 2:
                color = (0, 0, 255)  # Merah untuk "empty_pallet"
                part_info = {'No. ': nomor_box, 'Part Name': 'empty_pallet','Capacity': '6 Pcs', 'Remark': 'Need Re-Packing', 'RHD': '3Pcs', 'LHD': '3Pcs'}
            else:
                color = (255, 0, 0)  # Warna default jika label tidak dikenali
                part_info = {'No. ': nomor_box, 'Part No': 'N/A', 'Part Name': 'N/A', 'Stock Info': 'N/A'}

            # Tambahkan informasi bagian ke list data
            data.append(part_info)

            # Gambar bounding box di atas gambar
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)

            # Tambahkan nomor bounding box di atas bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3.0  # Ganti ukuran huruf di sini
            font_thickness = 2
            text_size = cv2.getTextSize(f'No. {nomor_box}', font, font_scale, font_thickness)[0]
            text_x = box[0] + (box[2] - box[0]) // 2 - text_size[0] // 2
            text_y = box[1] - 5
            cv2.putText(img, f'No. {nomor_box}', (text_x, text_y), font, font_scale, color, font_thickness)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konversi kembali ke RGB

        return img, data  # Mengembalikan gambar dan data


    def calculate_iou(box1, box2):
        """
        Menghitung Intersection over Union (IoU) antara dua bounding box.

        Parameters:
            box1 (list): Koordinat bounding box pertama [xmin, ymin, xmax, ymax].
            box2 (list): Koordinat bounding box kedua [xmin, ymin, xmax, ymax].

        Returns:
            float: Nilai IoU antara dua bounding box.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        area1 = w1 * h1
        area2 = w2 * h2

        intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0
        return iou

