from django.db import models

# Create your models here.


class CapturesImages(models.Model):
    image = models.FileField(upload_to='capture_images/' , null=True,blank=True)