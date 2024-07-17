import os
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
from pyzbar.pyzbar import decode
import base64
from io import BytesIO
from deepface import DeepFace
import cv2
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from werkzeug.utils import secure_filename
import uuid


class RequestError(Exception):
    """Exception raised for request errors."""

    def __init__(self, error_id):
        self.id = error_id

    @staticmethod
    def insufficient_data():
        return "RequestError: Insufficient data in request with id {self.id}"

    @staticmethod
    def wrong_file_type():
        return "RequestError: Wrong file type in request with id {self.id}"

    @staticmethod
    def wrong_request_method():
        return "RequestError: Wrong request method for request with id {self.id}"
    

UPLOAD_FOLDER = '/Users/lakshyadujari/Desktop/Projects/NIC_AI/QR/pythonProject/Image_Data' # Path to the folder where uploaded images will be saved and its path is saved in the database
ALLOWED_EXTENSIONS = {'PNG', 'JPG', 'JPEG', 'MPO'}  # Allowed file extensions
ALLOWED_IP_ADDR = {}  # Allowed IP addresses
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib",
          "SFace"]  # List of Face recognition models
backends = ['opencv', 'ssd', 'dlib', 'retinaface', 'yolov8', 'mediapipe']  # List of Face detection models
face_verification_threshold = 20  # Threshold for face verification
db_name = "User_Detail" # Postgres user database name
db_host = "localhost" # Postgres Database Host Name
db_user = "postgres" # Postgres user
db_pass = "Honey@1998" # Postgres Password


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# connection to the database
def connect_to_database():
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            host = db_host,
            database = db_name,
            user = db_user,
            password = db_pass
        )

        # Create a cursor object to interact with the database
        cur = conn.cursor()
        return conn, cur, True

    except psycopg2.errors.CannotConnectNow as e:
        print("Error in connecting with database", e)
        return None, None, False


# closing the connection
def close_connection(conn, cur):  # Connection and cursor object
    # Close the cursor and connection to the database
    cur.close()
    conn.close()
    return False


# function to create user table
def create_user_table(conn, cur,
                      open_connection=False):  # Connection, cursor object and weather to keep the connection open or not
    try:
        create_table_query = """
            CREATE TABLE IF NOT EXISTS user_info (
            "id"	SERIAL PRIMARY KEY,
            "name"	TEXT NOT NULL,
            "address"	TEXT,
            "phone"	TEXT,
            "email"	TEXT
            )
        """
        # Execute the SQL query
        cur.execute(create_table_query)

        # Commit the changes to the database
        conn.commit()
        user_table_check = True

        if not open_connection:
            close_connection(conn, cur)

        return user_table_check

    except Exception as e:
        conn.rollback()
        print("Error in creating user table ", e)
        user_table_check = False
        return user_table_check


# function to create embeddings table
def create_embeddings_table(conn, cur,
                            open_connection=False):  # Connection, cursor object and weather to keep the connection open or not
    try:
        # Define the SQL query to create the table
        create_table_query = """
            CREATE TABLE IF NOT EXISTS embeddings (
            "id" SERIAL PRIMARY KEY,
            "image"	TEXT NOT NULL,
            "user_id" INTEGER NOT NULL,
            "embedding" DECIMAL[] NOT NULL,
            FOREIGN KEY ("user_id") REFERENCES user_info ("id")
            )
        """
        # Execute the SQL query
        cur.execute(create_table_query)

        # Commit the changes to the database
        conn.commit()

        if not open_connection:
            close_connection(conn, cur)

        return True

    except Exception as e:
        conn.rollback()
        print("Error in creating user table ", e)
        return False


# function to creat the tables if not exists
def create_database_tables(conn, cur):  # Connection and cursor object
    user_table_check = create_user_table(conn, cur, open_connection=True)
    embeddings_table_check = create_embeddings_table(conn, cur, open_connection=True)
    if user_table_check and embeddings_table_check:
        return True
    else:
        return False


# function to fetching user record of single user
def fetch_single_usr_info(req_usr_id, confidence, conn, cur):  # requested user id, Connection and cursor object
    select_usr_query = """
    Select user_info.id,
        user_info.name,
        user_info.address,
        user_info.phone,
        user_info.email,
        embeddings.image,
        embeddings.user_id 
    from user_info 
    inner join embeddings on user_info.id = embeddings.user_id 
    where user_info.id = %s
    """
    cur.execute(select_usr_query, (req_usr_id,))
    result = cur.fetchone()
    close_connection(conn, cur)
    if result is not None:
        # image = pickle.loads(result[5])
        image = cv2.imread(result[5])
        image_base64 = base64.b64encode(image).decode('utf-8')
        return jsonify({'confidence': confidence,'user_id': result[0], 'name': result[1], 'address': result[2], 'phone': result[3],
                        'email': result[4], 'image': image_base64})
    else:
        return jsonify({'Error': 'User not found'}), 404


# function to fetch records of all the user_id passed in rows
def fetch_multiple_usr_info(rows, conn, cur):  # list of user_id, Connection and cursor object
    ids = [row[0] for row in rows]
    select_usr_query = """
    Select user_info.id,
        user_info.name,
        user_info.address,
        user_info.phone,
        user_info.email,
        embeddings.image,
        embeddings.user_id 
    from user_info 
    inner join embeddings on user_info.id = embeddings.user_id 
    where user_info.id = ANY(%s)
    """
    # Execute the query
    cur.execute(select_usr_query, (ids,))
    result = cur.fetchall()
    response = []
    if len(result) > 0:
        count = 0
        for user in result:
            # image = pickle.loads(user[5])
            image = cv2.imread(user[5])
            image_base64 = base64.b64encode(image).decode('utf-8')
            response.append({'confidence': rows[count][2],
                             'user_id': user[0],
                             'name': user[1],
                             'address': user[2],
                             'phone': user[3],
                             'email': user[4],
                             'image': (image_base64)})
            count += 1
        close_connection(conn, cur)
        return response
    close_connection(conn, cur)


# function to create embeddings of the image
def create_embeddings_array(image):  # image
    embedding = DeepFace.represent(image,
                                   model_name=models[2],
                                   detector_backend=backends[0],
                                   enforce_detection=False,
                                   align=True, normalization='Facenet2018')
    return embedding[0]['embedding']


# funnction to create a unique file name
def unique_filename(original_filename):
    base, ext = os.path.splitext(original_filename)
    unique = uuid.uuid4()
    return secure_filename(f"{unique}{ext}")


# function to create a plot of matched images with bbox
def matched_bbox(image1, image2, x1, y1, w1, h1, x2, y2, w2, h2,
                 probability):  # image1, image2, coordinates of bbox, probability of match
    # Define the coordinates
    # Draw the bounding box on img1
    cv2.rectangle(image1, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
    cv2.rectangle(image2, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

    # Display the image with the bounding box
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title("Matched with " + str(probability) + " %")
    plt.show()


# function to check for image is of correct extension
def allowed_file(filename):  # filename
    try:
        Image.open(filename).verify() # Checks if it's a valid image
        return Image.open(filename).format in ALLOWED_EXTENSIONS
    except (IOError, SyntaxError):
        return False


@app.route('/qr', methods=['POST'])
def qr_reader():
    req_id = None
    # currently printing the ip address of the client in future implement IP address filtering
    print(request.environ['REMOTE_ADDR'])
    try:
        # Checking if the request method is post
        if not request.method == 'POST':
            raise RequestError(0)

        # Checking if the request has all the required data
        if 'file' not in request.files:
            raise RequestError(1)

        # Checking if the file is of allowed extension
        if not allowed_file(request.files['file']):
            raise RequestError(2)

        # Getting data from request
        file = request.files['file']
        req_id = request.form.get('ID', type=int)

        # Processing the image
        image = Image.open(file).convert('RGB')
        draw = ImageDraw.Draw(image)
        bardata = []  # List to store all the barcodes data detected in the image

        # Getting all barcodes detected from the image
        for barcode in decode(image):
            rect = barcode.rect
            # drawing the rectangle around the barcode
            draw.rectangle(
                (
                    (rect.left - 10, rect.top - 10),
                    (rect.left + rect.width + 10, rect.top + rect.height + 10)
                ),
                outline='#0080ff',
                width=5
            )
            draw.polygon(barcode.polygon, outline='#e945ff')
            bardata.append({
                'data': barcode.data.decode('utf-8'),
                'type': barcode.type
            })

        # Preparing the response
        if bardata.__len__() > 0:
            # Encode the image as base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            response_data = {'Id': req_id, 'status_code': 100, 'barData': bardata, 'image': encoded_image}
            return jsonify(response_data), 201

        else:
            response_data = {'Id': req_id, 'status_code': 99}
            return jsonify(response_data), 206

    # Handling the exceptions
    except (RequestError,Exception) as e:
        if isinstance(e, RequestError):
            if e.id == 2:
                return jsonify({'Id': req_id, 'status_code': 404, 'error': e.wrong_file_type()}), 404
            elif e.id == 1:
                return jsonify({'Id': req_id, 'status_code': 404, 'error': e.insufficient_data()}), 404 
            elif e.id == 0: 
                return jsonify({'Id': req_id, 'status_code': 404, 'error': e.wrong_request_method()}), 404
        else:
            return jsonify({'Id': req_id,
                            'error': f'Error in Processing Image',
                            'status_code': 98,
                            'description': str(e)
                            }), 404


@app.route('/facial_recognition', methods=['POST'])
def face_detection():
    conn = None
    cur = None
    req_id = None
    print(request.environ[
              'REMOTE_ADDR'])  # currently printing the ip address of the client in future implement IP address filtering
    try:
        # Checking if the request method is post
        if not request.method == 'POST':
            raise RequestError(0)

        # Checking if the request has all the required data
        if not 'img' in request.files:
            raise RequestError(1)

        # checking if the file is of allowed extension
        if not allowed_file(request.files['img']):
            raise RequestError(2)

        # Connecting to the database
        conn, cur, database_check = connect_to_database()
        if not database_check:
            raise psycopg2.errors.CannotConnectNow

        # fetching the request id
        req_id = request.form.get('ID', type=int)
        # getting image
        target_img = Image.open(request.files['img'])
        target_img = np.array(target_img)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        # Getting the embeddings of the image
        target = create_embeddings_array(target_img)

        # Check if all tables exists
        if create_database_tables(conn, cur):
            # from the embeddings of the target image calculating distance of all the images in the database and filtering based of verification threshold
            # query = f"""
            #     select user_id,image, distance
            #     from (
            #         select user_id,image, sqrt(sum(distance)) as distance
            #         from (
            #             select user_id,image, pow(unnest(embedding) - unnest(ARRAY{target}), 2) as distance
            #             from embeddings
            #         ) sq
            #         group by user_id, image
            #     ) sq2
            #     where distance < {face_verification_threshold}
            #     order by distance
            # """
            # Query which will get the record with minimum result what ever it is
            query = f"""
                select user_id,image, min(distance) as distance
                from (
                    select user_id,image, sqrt(sum(distance)) as distance
                    from (
                        select user_id,image, pow(unnest(embedding) - unnest(ARRAY{target}), 2) as distance
                        from embeddings
                    ) sq
                    group by user_id, image
                ) sq2
                group by user_id, image
                order by distance 
                LIMIT 1
            """
            cur.execute(query)
            rows = cur.fetchall()
            if int(rows[0][2]) > face_verification_threshold:
                print('greater than threshold')
            # Getting information of the detected users
            if len(rows) == 1:
                return fetch_single_usr_info(rows[0][0], rows[0][2], conn, cur)
            elif len(rows) > 1:
                return fetch_multiple_usr_info(rows, conn, cur)
            else:
                return jsonify({'verified': False})

    # Handling the exceptions
    except (psycopg2.errors.CannotConnectNow, psycopg2.DatabaseError,RequestError, Exception) as e:
        if conn is not None:
            conn.rollback()
            close_connection(conn, cur)
        if isinstance(e, RequestError):
            if e.id == 2:
                return jsonify({'Id': req_id, 'status_code': 404, 'error': e.wrong_file_type()}), 404
            elif e.id == 1:
                return jsonify({'Id': req_id, 'status_code': 404, 'error': e.insufficient_data()}), 404 
            elif e.id == 0: 
                return jsonify({'Id': req_id, 'status_code': 404, 'error': e.wrong_request_method()}), 404
        else:
            return jsonify({'Id': req_id,
                            'error': f'Error in Processing Image',
                            'status_code': 98,
                            'description': str(e)
                            }), 404


@app.route('/insert_usr_record', methods=['POST'])
def insert_user_record():
    conn = None
    cur = None
    file_path = None
    try:
        # checking if the request method is post
        if not request.method == 'POST':
            raise RequestError(0)

        # checking if the request has all the required data
        if not ('img' in request.files and
                'Name' in request.form and
                'Email' in request.form and
                'Phone' in request.form and
                'Address' in request.form):
            raise RequestError(1)

        # checking if the file is of allowed extension
        if not allowed_file(request.files['img']):
            raise RequestError(2)

        # connecting to the database
        conn, cur, databasecheck = connect_to_database()

        # checking if the database is connected
        if not databasecheck:
            raise psycopg2.errors.CannotConnectNow

        # creating the tables if not exists
        if not create_database_tables(conn, cur):
            raise psycopg2.errors.UndefinedTable

        # saving image to disk
        file = request.files['img']
        file = Image.open(file).convert('RGB')
        filename = unique_filename(request.files['img'].filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # if the folder does not exists then create the folder
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        file.save(file_path)
        
        # Getting all the required data from the request
        req_id = request.form.get('ID', type=int)
        name = request.form.get('Name', type=str)
        email = request.form.get('Email', type=str)
        phone = request.form.get('Phone', type=str)
        address = request.form.get('Address', type=str)
        
        # converting image to ndarray so that it can be used for creating embeddings
        image = Image.open(request.files['img'])
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # creating feature set of images
        embedding = create_embeddings_array(image)
        # image = pickle.dumps(image)
        
        # Getting the last value of the user_id
        cur.execute("Select count(id) from user_info")
        result = cur.fetchone()
        
        if result is not None:
            next_id = result[0] + 1
            # Define the SQL query to insert a record in user tabel
            insert_user_query = """
                INSERT INTO user_info ("name", "address", "phone", "email")
                VALUES (%s, %s, %s, %s)
            """
            # Execute the SQL query with the record values
            cur.execute(insert_user_query, (name, address, phone, email))

            # Defining the sql query to insert record in embeddings tabel
            insert_embedding_query = """
                INSERT INTO embeddings ("image", "user_id", "embedding")
                VALUES (%s, %s, %s)
            """
            cur.execute(insert_embedding_query, (file_path, next_id, embedding))
            conn.commit()
            
            # closing the connection
            close_connection(conn, cur)

            # returning the response
            return (jsonify({'status_code': 201, 'request_id': req_id, 'message': 'Record Inserted'}),
                    201)
        else:
            # returning the response
            close_connection(conn, cur)
            return jsonify({'status_code': 404, 'error': 'Cannot Get unique ID'}), 404

    except (TypeError,
            ValueError,
            psycopg2.errors.UndefinedTable,
            psycopg2.errors.CannotConnectNow,
            psycopg2.Error,
            psycopg2.DatabaseError,
            RequestError,
            Exception) as e:
        if conn is not None:
            conn.rollback()
            close_connection(conn, cur)
        if file_path is not None:
            os.remove(file_path)
        if isinstance(e, RequestError):
            if e.id == 2:
                return jsonify({'Id': req_id, 'status_code': 404, 'error': e.wrong_file_type()}), 404
            elif e.id == 1:
                return jsonify({'Id': req_id, 'status_code': 404, 'error': e.insufficient_data()}), 404 
            elif e.id == 0: 
                return jsonify({'Id': req_id, 'status_code': 404, 'error': e.wrong_request_method()}), 404
        else:
            return jsonify({'status_code': 404, 'error': "Error from Database" + str(e)}), 404


@app.route('/get_usr_data', methods=['POST'])
def get_usr_data():
    try:
        # Checking if the request method is post
        if not request.method == 'POST':
            raise RequestError(0)

        # Checking if the request has all the required data
        if not 'id' in request.form:
            raise RequestError(1)

        # Connecting to the database
        conn, cur, check = connect_to_database()

        # checking if the database is connected
        if not check:
            raise psycopg2.errors.CannotConnectNow

        req_usr_id = request.form['id']
        return fetch_single_usr_info(req_usr_id, conn, cur)

    except (psycopg2.errors.CannotConnectNow,RequestError, Exception) as e:
        if e is isinstance(e,RequestError):
            if e.id == 2:
                return jsonify({'Id': 1, 'status_code': 404, 'error': e.wrong_file_type()}), 404
            elif e.id == 1:
                return jsonify({'Id': 1, 'status_code': 404, 'error': e.insufficient_data()}), 404 
            elif e.id == 0: 
                return jsonify({'Id': 1, 'status_code': 404, 'error': e.wrong_request_method()}), 404
        else:
            return jsonify({'status_code': 404, 'error': "Error from Database" + str(e)}), 404


if __name__ == '__main__':
    app.run(debug=False, port=8080)
    