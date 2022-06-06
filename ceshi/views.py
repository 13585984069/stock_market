from django.shortcuts import render

# Create your views here.
def ceshi(request):
    return render(request, 'block.html')