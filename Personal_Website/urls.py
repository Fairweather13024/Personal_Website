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
from django.contrib.auth.views import LogoutView
from django.urls import include, path

#User auth
from django.conf.urls import url
from django.contrib.auth import views as auth_views

import Personal_Website
from my_personal_portfolio import views

from django.contrib import admin
from django.urls import include, path
from django.contrib.auth.views import LogoutView

import Personal_Website

urlpatterns = [
    #path('', include(Personal_Website.urls)),
    path('', views.index),
    path('home/', views.index, name = 'home'),
    path('projects/', views.projects, name = 'projects'),
    path('profile/', views.profile_page),

    path('login/', views.log_in, name = 'login'),
    path('admin/', admin.site.urls),
    path('sign_up/', views.registration, name = "sign_up"),
    path('stock/', views.stock, name = 'stock'),
    path('logout/', LogoutView.as_view(), name='logout'),

    path('accounts/', include('django.contrib.auth.urls')),  # resetting passwords
    path('password_reset/done/',
         auth_views.PasswordResetDoneView.as_view(template_name='password/password_reset_done.html'),
         name='password_reset_done'),
    path('reset/<uidb64>/<token>/',
         auth_views.PasswordResetConfirmView.as_view(template_name="password/password_reset_confirm.html"),
         name='password_reset_confirm'),
    path('reset/done/',
         auth_views.PasswordResetCompleteView.as_view(template_name='password/password_reset_complete.html'),
         name='password_reset_complete'),
    path("password_reset", views.password_reset_request, name="password_reset"),


]
