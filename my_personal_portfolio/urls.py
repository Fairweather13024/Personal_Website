#i created this
"""
Defines URL patterns for my_personal_portfolio

"""

from django.urls import path
from my_personal_portfolio.migrations import views

app_name = 'my_personal_website'

urlpatterns = [
    path('', views.template, name = 'template'),

]