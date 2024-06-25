import os
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import urllib.request

# Define the path to the model
model_path = "best-cataract-od.pt"

# Function to download the model if not present
def download_model(model_path):
    if not os.path.exists(model_path):
        url = "blob:https://github.com/0b5a1165-ca8b-4d4a-a846-b696279e008e"  # Replace with your model file URL
        urllib.request.urlretrieve(url, model_path)
        st.write(f"Model downloaded from {url}")

# Ensure the model is downloaded
download_model(model_path)

# Load YOLOv8 model
model = YOLO(model_path)

# Function to perform prediction
def predict_image(input_image):
    # Convert Streamlit input image (PIL Image) to numpy array
    image_np = np.array(input_image)

    # Ensure the image is in the correct format
    if len(image_np.shape) == 2:  # grayscale to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # RGBA to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    # Perform prediction
    results = model(image_np)

    # Draw bounding boxes on the image
    image_with_boxes = image_np.copy()
    raw_predictions = []
    for result in results[0].boxes:
        label = "Normal" if result.cls.item() == 1 else "Cataract"  # Convert tensor to standard Python type
        confidence = result.conf.item()  # Convert tensor to standard Python type
        xmin, ymin, xmax, ymax = map(int, result.xyxy[0])
        cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        
        # Enlarge font scale and thickness
        font_scale = 1.0
        thickness = 2
        cv2.putText(image_with_boxes, f'{label} {confidence:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        raw_predictions.append(f"Label: {label}, Confidence: {confidence:.2f}, Box: [{xmin}, {ymin}, {xmax}, {ymax}]")

    raw_predictions_str = "\n".join(raw_predictions)

    return image_with_boxes, raw_predictions_str

# Streamlit interface
st.title("YOLOv8 Cataract Screener")
st.write("Nexus-Health Lite V.1.0.0")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    input_image = Image.open(uploaded_image)
    predicted_image, raw_result = predict_image(input_image)
    
    st.image(predicted_image, caption='Predicted Image')
    st.text_area("Raw Result", raw_result)
