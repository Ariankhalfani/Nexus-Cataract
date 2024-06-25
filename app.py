import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import sqlite3
import streamlit as st
from io import BytesIO

# Load YOLO models
try:
    yolo_model_cataract = YOLO('best-cataract.pt')
    yolo_model_detection = YOLO('best-cataract-od.pt')
    st.write("YOLO models loaded successfully.")
except Exception as e:
    st.write(f"Error loading YOLO models: {e}")

def calculate_ratios(red_values, green_values, blue_values, total_pixels):
    if total_pixels == 0:
        return 0, 0, 0

    red_ratio = np.sum(red_values) / total_pixels
    green_ratio = np.sum(green_values) / total_pixels
    blue_ratio = np.sum(blue_values) / total_pixels

    total_ratio = red_ratio + green_ratio + blue_ratio

    if total_ratio > 0:
        red_quantity = (red_ratio / total_ratio) * 255
        green_quantity = (green_ratio / total_ratio) * 255
        blue_quantity = (blue_ratio / total_ratio) * 255
    else:
        red_quantity, green_quantity, blue_quantity = 0, 0, 0

    return red_quantity, green_quantity, blue_quantity

def cataract_staging(red_quantity, green_quantity, blue_quantity):
    average_rgb = (red_quantity + green_quantity + blue_quantity) / 3
    
    if average_rgb < 85:
        stage = "Incipient"
    elif average_rgb < 170:
        stage = "Immature"
    elif average_rgb < 255:
        stage = "Mature"
    else:
        stage = "Hypermature"
    
    return stage

def predict_and_visualize(image):
    try:
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        orig_size = pil_image.size
        results = yolo_model_cataract(pil_image)

        raw_response = str(results)
        masked_image = np.array(pil_image)
        mask_image = np.zeros_like(masked_image)

        red_quantity, green_quantity, blue_quantity = 0, 0, 0
        total_pixels = 0

        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                mask = np.array(result.masks.data.cpu().squeeze().numpy())
                mask_resized = np.array(Image.fromarray(mask).resize(orig_size, Image.NEAREST))

                red_mask = np.zeros_like(masked_image)
                red_mask[mask_resized > 0.5] = [255, 0, 0]
                alpha = 0.5
                blended_image = cv2.addWeighted(masked_image, 1 - alpha, red_mask, alpha, 0)

                pupil_pixels = np.array(pil_image)[mask_resized > 0.5]
                total_pixels = pupil_pixels.shape[0]

                red_values = pupil_pixels[:, 0]
                green_values = pupil_pixels[:, 1]
                blue_values = pupil_pixels[:, 2]

                red_quantity, green_quantity, blue_quantity = calculate_ratios(red_values, green_values, blue_values, total_pixels)
                stage = cataract_staging(red_quantity, green_quantity, blue_quantity)

                # Add text to the blended image
                combined_pil_image = Image.fromarray(blended_image)
                draw = ImageDraw.Draw(combined_pil_image)
                
                # Load a larger font (adjust the size as needed)
                font_size = 48  # Example font size
                try:
                    font = ImageFont.truetype("font.ttf", size=font_size)
                except IOError:
                    font = ImageFont.load_default()
                    st.write("Error: cannot open resource, using default font.")

                text = f"Red quantity: {red_quantity:.2f}\nGreen quantity: {green_quantity:.2f}\nBlue quantity: {blue_quantity:.2f}\nStage: {stage}"
                
                # Calculate text bounding box
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                text_x = 20
                text_y = 40
                padding = 10

                # Draw a filled rectangle for the background
                draw.rectangle(
                    [text_x - padding, text_y - padding, text_x + text_width + padding, text_y + text_height + padding],
                    fill="black"
                )
                
                # Draw text on top of the rectangle
                draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)

                return np.array(combined_pil_image), red_quantity, green_quantity, blue_quantity, raw_response, stage

        return image, 0, 0, 0, "No mask detected.", "Unknown"
    
    except Exception as e:
        st.write("Error:", e)
        return np.zeros_like(image), 0, 0, 0, str(e), "Error"

def create_connection(db_file):
    """ Create a database connection to the SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        st.write(e)
    return conn

def create_cataract_table(conn):
    """ Create the cataract results table if it does not exist """
    create_table_sql = """ CREATE TABLE IF NOT EXISTS cataract_results (
                            id integer PRIMARY KEY,
                            image blob,
                            red_quantity real,
                            green_quantity real,
                            blue_quantity real,
                            stage text
                        ); """
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
    except sqlite3.Error as e:
        st.write(e)

def check_duplicate_entry(conn, red_quantity, green_quantity, blue_quantity, stage):
    cursor = conn.cursor()
    query = '''SELECT COUNT(*) FROM cataract_results WHERE red_quantity=? AND green_quantity=? AND blue_quantity=? AND stage=?'''
    cursor.execute(query, (red_quantity, green_quantity, blue_quantity, stage))
    count = cursor.fetchone()[0]
    return count > 0

def save_cataract_prediction_to_db(image, red_quantity, green_quantity, blue_quantity, stage):
    database = "cataract_results.db"
    conn = create_connection(database)
    if conn:
        create_cataract_table(conn)
        
        # Check for duplicate entries
        if check_duplicate_entry(conn, red_quantity, green_quantity, blue_quantity, stage):
            conn.close()
            return "Duplicate entry found, not saving.", "Duplicate entry detected."
        
        sql = '''INSERT INTO cataract_results(image, red_quantity, green_quantity, blue_quantity, stage) VALUES(?,?,?,?,?)'''
        cur = conn.cursor()
        
        # Convert the image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        cur.execute(sql, (img_bytes, red_quantity, green_quantity, blue_quantity, stage))
        conn.commit()
        conn.close()
        return "Data saved successfully", f"Red: {red_quantity}, Green: {green_quantity}, Blue: {blue_quantity}, Stage: {stage}"

    return "Failed to save data", "No connection to the database."

def combined_prediction(image):
    blended_image, red_quantity, green_quantity, blue_quantity, raw_response, stage = predict_and_visualize(image)
    save_message, debug_info = save_cataract_prediction_to_db(Image.fromarray(blended_image), red_quantity, green_quantity, blue_quantity, stage)
    return blended_image, red_quantity, green_quantity, blue_quantity, raw_response, stage, save_message, debug_info

def download_database():
    database = "cataract_results.db"
    with open(database, 'rb') as f:
        st.download_button("Download Database", f, file_name=database)

def predict_object_detection(image):
    # Perform prediction with YOLO object detection model
    results = yolo_model_detection(image)
    # Draw bounding boxes on the image
    image_with_boxes = image.copy()
    for result in results[0].boxes:
        label = "Normal" if result.cls.item() == 1 else "Cataract"
        confidence = result.conf.item()
        xmin, ymin, xmax, ymax = map(int, result.xyxy[0])
        cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(image_with_boxes, f'{label} {confidence:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image_with_boxes

st.title("Cataract Screener and Analyzer")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    input_image = Image.open(uploaded_image)
    input_np_image = np.array(input_image)
    
    # Display the uploaded image
    st.image(input_np_image, caption='Uploaded Image', use_column_width=True)
    
    # Predict and visualize
    if st.button('Submit'):
        detection_image = predict_object_detection(input_np_image)
        analyzer_image, red_quantity, green_quantity, blue_quantity, raw_response, stage, save_message, debug_info = combined_prediction(input_np_image)
        
        # Display results
        st.image(detection_image, caption='Object Detection Image', use_column_width=True)
        st.image(analyzer_image, caption='Cataract Analyzer Image', use_column_width=True)
        
        st.write(f"Red Quantity: {red_quantity:.2f}")
        st.write(f"Green Quantity: {green_quantity:.2f}")
        st.write(f"Blue Quantity: {blue_quantity:.2f}")
        st.write(f"Cataract Stage: {stage}")
        st.write(f"Raw Response: {raw_response}")
        st.write(f"Database Save Message: {save_message}")
        st.write(f"Debug Info: {debug_info}")

# Add button to download database
download_database()
