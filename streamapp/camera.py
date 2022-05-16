import self as self
from tensorflow._api.v2.config import threading
import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import threading
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import mysql.connector


class VideoCamera(object):

    def __init__(self):

        self.video = cv2.VideoCapture(0)
        # cv2.namedWindow("preview")
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):

        cwd = os.getcwd()
        emotions_array = []
        model = model_from_json(open(f'{cwd}/trained_model/fer.json', "r").read())
        # # load weights
        model.load_weights(f'{cwd}/trained_model/fer.h5')
        name_fc = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'  # getting a haarcascade xml file
        fc = cv2.CascadeClassifier()  # processing it for our project
        if not fc.load(cv2.samples.findFile(name_fc)):  # add an error check
            print("Error loading xml file")
        while True:
            (self.grabbed, self.frame) = self.video.read()
            frame = self.frame
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale to facilitate analysis
            face = fc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for x, y, w, h in face:
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                roi_gray = gray[y:y + w, x:x + h]  # cropping the face
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255
                predictions = model.predict(img_pixels)

                try:
                    max_index = np.argmax(predictions[0])
                    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                    predicted_emotion = emotions[max_index]
                    print(max_index)  # print out the dominant emotion
                    emotion_list = str(max_index)
                    print(emotion_list)

                except:
                    print("no face")


class kill_video():



    def insert_varibles_into_table(user_id, emotions, date):

            connection = mysql.connector.connect(host='database-2.cjyi04obosa8.eu-west-1.rds.amazonaws.com',
                                                     database='mydb2',
                                                     user='admin',
                                                     password='Mood2018257')
            cursor = connection.cursor()
            mySql_insert_query = """INSERT INTO Mood ( Auth_user_id, Emotions, Date) 
                                                                         VALUES (%s, %s, %s) """

            record = (user_id, emotions, date)
            cursor.execute(mySql_insert_query, record)
            connection.commit()
            print("Record inserted successfully into Mood table")

    insert_varibles_into_table(1, 23232323, '20')









