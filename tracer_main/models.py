from django.db import models

class Canvas(models.Model):
    name = models.CharField(max_length=100, default="")
    image = models.BinaryField()
