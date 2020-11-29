from django.db import models

# Create your models here.
class UserRegistration(models.Model):
    Email = models.EmailField()
    Password = models.CharField(max_length= 50)

    def __str__(self):
        return self.UserName