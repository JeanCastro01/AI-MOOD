from django.urls import path, include, re_path
from apps.home import views


urlpatterns = [

    path('', views.index, name='index'),

    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),

   #path('video_feed', views.video_feed, name='video_feed'),

]
