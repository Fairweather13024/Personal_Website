from django.shortcuts import render

# Create your views here.
def template(request):
    return render(request, 'my_personal_portfolio/template.html')

def projects(request):
    return render(request, 'my_personal_portfolio/projects.html')

def profile_page(request):
    return render(request, 'my_personal_portfolio/profile-page.html')

def log_in(request):
    return render(request, 'my_personal_portfolio/login-page.html')

