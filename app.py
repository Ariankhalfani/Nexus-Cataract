import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLOv8 model
model = YOLO("best-cataract-od.pt")

# Function to perform prediction
def predict_image(input_image):
    # Convert Gradio input image (PIL Image) to numpy array
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

# Create Gradio interface
inputs = gr.Image(type="pil")
outputs = [gr.Image(type="numpy", label="Predicted Image"), gr.Textbox(label="Raw Result")]
title = "YOLOv8 Cataract Screener"
description = "Nexus-Health Lite V.1.0.0"
iface = gr.Interface(fn=predict_image, inputs=inputs, outputs=outputs, title=title, description=description)

# Launch the interface
iface.launch()
