import streamlit as st
from PIL import Image
import numpy as np
from cataract import combined_prediction, predict_object_detection
import sqlite3
import base64

def download_database():
    database = "cataract_results.db"
    with open(database, 'rb') as f:
        st.download_button("Download Database", f, file_name=database)

def fetch_all_data(conn, table_name):
    """ Fetch all data from the given table """
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    return rows

def get_db_data(db_path_cataract):
    conn_cataract = sqlite3.connect(db_path_cataract)
    if conn_cataract:
        create_table(conn_cataract)
        cataract_data = fetch_all_data(conn_cataract, "cataract_results")
        conn_cataract.close()
        return cataract_data
    else:
        return []

def format_db_data(cataract_data):
    """ Format the database data for display """
    formatted_data = ""

    if not cataract_data:
        return "No data available in the database."

    headers = ["ID", "Image", "Red Quantity", "Green Quantity", "Blue Quantity", "Stage"]
    formatted_data += "<h2>Cataract Data</h2><table border='1'><tr>" + "".join([f"<th>{header}</th>" for header in headers]) + "</tr>"

    for row in cataract_data:
        image_html = "No image"
        if row[1] is not None:
            image = base64.b64encode(row[1]).decode('utf-8')
            image_html = f"<img src='data:image/png;base64,{image}' width='100'/>"

        formatted_data += "<tr>" + "".join([f"<td>{image_html if i == 1 else row[i]}</td>" for i in range(len(row))]) + "</tr>"

    formatted_data += "</table>"

    return formatted_data

def create_table(conn):
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
        print(e)

# Streamlit app
st.title("Cataract Detection")

st.sidebar.title("Options")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Upload and Predict", "View Database", "Download Database"])

if app_mode == "Upload and Predict":
    st.header("Upload an Eye Image")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        input_np_image = np.array(image)
        analyzer_image, red_quantity, green_quantity, blue_quantity, raw_response, stage, save_message = combined_prediction(input_np_image)
        
        st.image(analyzer_image, caption='Processed Image.', use_column_width=True)
        st.write(f"Red Quantity: {red_quantity:.2f}")
        st.write(f"Green Quantity: {green_quantity:.2f}")
        st.write(f"Blue Quantity: {blue_quantity:.2f}")
        st.write(f"Stage: {stage}")
        st.write(f"Raw Response: {raw_response}")
        st.write(f"Save Message: {save_message}")
        
        detection_image = predict_object_detection(input_np_image)
        st.image(detection_image, caption='Object Detection Image.', use_column_width=True)

elif app_mode == "View Database":
    st.header("Cataract Database")
    database = "cataract_results.db"
    cataract_data = get_db_data(database)
    formatted_data = format_db_data(cataract_data)
    st.markdown(formatted_data, unsafe_allow_html=True)

elif app_mode == "Download Database":
    st.header("Download Database")
    download_database()
