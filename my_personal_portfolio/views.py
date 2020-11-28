from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, 'index.html')
def template(request):
    return render(request, 'template.html')

def projects(request):
    return render(request, 'projects.html')

def profile_page(request):
    return render(request, 'profile-page.html')

def log_in(request):
    return render(request, 'login-page.html')

