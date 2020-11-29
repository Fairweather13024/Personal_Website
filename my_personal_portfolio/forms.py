from django import forms
from .models import People

class Signup(forms.ModelForm):
    class Meta:
        model = People
        fields = ['email', 'password']
        widgets = {
            'email': forms.TextInput(attrs={
                'class': 'form-control',
                'maxlength': 256,
                'placeholder': "Email...",
                'type': 'email'
            }),
            'password': forms.PasswordInput(attrs={
                'class': 'form-control',
                'maxlength': 50,
                'placeholder': "Password..."
            })
        }

class Login(forms.ModelForm):
    class Meta:
        model = People
        fields = ['email', 'password']
        widgets = {
            'email': forms.TextInput(attrs={
                'class': 'form-control',
                'maxlength': 256,
                'placeholder': "Email...",
                'type': 'email'
            }),
            'password': forms.PasswordInput(attrs={
                'class': 'form-control',
                'maxlength': 50,
                'placeholder': "Password..."
            })
        }