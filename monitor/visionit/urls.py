from django.urls import path

from . import views

urlpatterns = [
    path('map', views.map, name='map'),
    path('state', views.is_busy, name='state'),
    path('rename/?P<prev_name>', views.change_name, name='renamepost'),
    path('delete/?P<name>', views.delete_name, name='deletepost'),
    path('rename', views.rename, name='rename'),
    path('lang/?P<lang>', views.lang_detect, name='lang'),
    path('', views.index, name='index'),
]
