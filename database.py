import sqlite3
from io import BytesIO

def create_connection(db_file):
    """ Create a database connection to the SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
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
        print(e)

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
            return "Duplicate entry found, not saving."
        
        sql = '''INSERT INTO cataract_results(image, red_quantity, green_quantity, blue_quantity, stage) VALUES(?,?,?,?,?)'''
        cur = conn.cursor()
        
        # Convert the image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        cur.execute(sql, (img_bytes, red_quantity, green_quantity, blue_quantity, stage))
        conn.commit()
        conn.close()
        return "Data saved successfully."

    return "Failed to save data."
