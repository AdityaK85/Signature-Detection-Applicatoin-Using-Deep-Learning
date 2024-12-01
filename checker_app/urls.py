from django.urls import path
from .views import *
from django.conf import settings
from django.conf.urls.static import static


ajax_url = [

    path('', index),
    path('compare_sigature', compare_sigature)

] 


urlpatterns = [ *ajax_url ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)