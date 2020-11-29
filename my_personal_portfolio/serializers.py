from django.contrib.auth.models import User, Group
from rest_framework import serializers
from my_personal_portfolio.models import UserRegistration


class UserRegistrationSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = UserRegistration
        fields = ['Email', 'Password']

