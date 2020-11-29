from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .forms import Signup, Login
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages

# Create your views here.
def index(request):
    return render(request, 'index.html')
def template(request):
    return render(request, 'template.html')

def projects(request):
    return render(request, 'projects.html')

def profile_page(request):
    return render(request, 'profile-page.html')

@login_required
def success(request):
    return render(request, 'stock.html')

def log_in(request):
    if request.user.is_authenticated:
        return redirect('stock')

    context = {}
    form = Login(request.POST or None)

    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(username=email, password=password)

        if user is not None:
            # correct username and password login the user
            login(request, user)
            return render(request, 'stock.html')

        else:
            context['message'] = "Sorry. You got your email or password wrong."

    context['form'] = form
    return render(request, 'accounts/login-page.html', context)

def registration(request):
    context = {}
    form = UserCreationForm(request.POST or None)

    if request.method == "POST":
        if form.is_valid():
            user = form.save()
            login(request, user)
            context['message'] = "Signup successful. Login here."
            return redirect('stock')
        else:
            context['message'] = "Play around with your password a bit more & try again."

    context['form'] = form
    return render(request, 'accounts/sign_up.html', context)