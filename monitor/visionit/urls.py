from django.urls import path

from . import views

urlpatterns = [
    path('map', views.map, name='map'),
    path('rename/?P<prev_name>', views.change_name, name='renamepost'),
    path('rename', views.rename, name='rename'),
    path('', views.index, name='index'),
]