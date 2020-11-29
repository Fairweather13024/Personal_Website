from django.db import models

# Create your models here.
class People(models.Model):
    email = models.EmailField()
    password = models.CharField(max_length= 50)
    last_login = models.TextField(null=True)

    def __str__(self):
        return self.email