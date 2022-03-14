from django.urls import path
from . import views
from .views import list_users

urlpatterns = [
    #path('', views.IndexView.as_view(), name='index'),
    path('',list_users)
]