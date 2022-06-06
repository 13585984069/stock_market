from django.shortcuts import render, redirect
from django.contrib.auth.models import User, auth
from django.contrib import messages

# Create your views here.


def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        if password2 == password1:
            if User.objects.filter(username=username).exists():
                messages.info(request, 'Username Taken')
                return redirect('account:register')
            else:
                user = User.objects.create_user(username=username, password=password1, email=email)
                user.save()
                return redirect('/')
        else:
            messages.info(request, 'Two passwords do not match')
            return redirect('account:register')
    else:
        return render(request, 'register.html')

def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password1 = request.POST['password1']
        user = auth.authenticate(username=username, password=password1)
        if user is not None:
            auth.login(request, user)
            return redirect('/stock/', {'user': user})
        else:
            messages.info(request, 'User is not exist or password is wrong')
            return redirect('account:login')
    else:
        return render(request, 'login.html')

def logout(request):
    auth.logout(request)
    return redirect('/')


