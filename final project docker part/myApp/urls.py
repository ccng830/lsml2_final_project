from django.contrib import admin
from django.urls import path, include
from . import views
from django.conf.urls import handler404, handler500
from django.conf.urls.static import static
from django.conf import settings
# 將我們剛剛設定的 View 引入


app_name = 'app'

urlpatterns = [
    path('', views.index),
    path('hello/', views.hello)
    ]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)