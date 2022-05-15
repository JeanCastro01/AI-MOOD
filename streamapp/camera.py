import pandas as pd
import cv2
import os
import numpy as np
from tensorflow._api.v2.config import threading
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from django.http import HttpResponse
from django.shortcuts import render
from .models import *
from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
#load model
from core import settings

#load model
from core import settings

#model = model_from_json(open("../trained_model/fer.json", "r").read())

#model = model_from_json(os.path.join(settings.BASE_DIR,'../trained_model/fer.json','r'))

#load weights
#model.load_weights('../trained_model/fer.h5')

#face_cascade_name = cv2.data.haarcascades + 'opencv_haarcascade_data/haarcascade_frontalface_alt.xml'  #getting a haarcascade xml file

#face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'

#face_cascade_name = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR,'haarcascade_frontalface_alt.xml'))
#face_cascade = cv2.CascadeClassifier()  #processing it for our project


#if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  #adding a fallback event
   # print("Error loading xml file")
#emotions_array = []



class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()
