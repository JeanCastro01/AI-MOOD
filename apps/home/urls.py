# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from apps.home import views
from streamapp import views

urlpatterns = [

    # The home page
    path('', views.index, name='home'),

    # Matches any html file
    #re_path(r'^.*\.*', views, name='pages'),

    path('video_feed', views.video_feed, name='video_feed'),
    path('charts', views.charts, name='charts'),
    path('kill_video', views.kill_video, name='kill_video'),

]
