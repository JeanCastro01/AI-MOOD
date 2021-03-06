import cv2
import datetime
from django.shortcuts import render
from django.http.response import StreamingHttpResponse, HttpResponse
from streamapp.camera import VideoCamera, kill_video


# Create your views here.


def index(request):
    return render(request, 'home/index.html')


def charts(request):
    return render(request, 'home/charts-morris.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def kill_video(request):
    return StreamingHttpResponse(gen(kill_video()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

