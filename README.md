# PlateX
PlateX – AI License Plate Detection & Recognition System

Description:
PlateX is an AI-driven computer vision project that automatically detects and recognizes vehicle license plates from images. The system integrates YOLOv12 for object detection and PaddleOCR for text extraction, achieving accurate and real-time performance. It includes a custom post-processing module to format Sri Lankan-style license plates and extract vehicle details such as province, category, and fuel type.

The project was trained using the Roboflow License Plate Dataset and implemented in Python within the Google Colab environment. A Flask web interface allows users to upload vehicle images, visualize detections, and export recognized data to a CSV file for further analysis.

Key Technologies:
Python | YOLOv12 | PaddleOCR | Flask | OpenCV | Roboflow | Google Colab

Features:

Real-time license plate detection and OCR recognition

Automatic plate text formatting and validation

Vehicle data extraction (province, category, fuel type)

Image upload via Flask web interface

CSV-based record storage

Deployable as a local or containerized web app
