"""Personal_Website URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from django.contrib.auth.views import LogoutView

import Personal_Website
from my_personal_portfolio import views


urlpatterns = [
    #path('', include(Personal_Website.urls)),

    path('', views.index),
    path('home/', views.index, name = 'home'),
    path('projects/', views.projects, name = 'projects'),
    path('profile/', views.profile_page),

    path('login/', views.log_in, name = 'log-in'),
    path('admin/', admin.site.urls),
    path('sign_up/', views.registration, name = "sign_up"),
    path('stock/', views.success, name = 'stock'),
    path('logout/', LogoutView.as_view(), name='logout'),
]
