from django.db import models

# Create your models here.

#class User(models.Model):
#	sname = models.CharField()
#	sgender = models.BooleanField(default=True)
#	sage = models.IntegerField()
#	isDelete = models.BooleanField(default=True)
#	sgrade = models.ForeignKey("Grades", on_delete=models.DO_NOTHING)
#
from django.db import models
from django.utils import timezone

class Photo(models.Model):
    image = models.ImageField(upload_to='image/', blank=False, null=False)
    upload_date = models.DateField(default=timezone.now)
