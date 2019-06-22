import os

from django.shortcuts import render
from django.template import loader
import json

from monitor import settings


def index(request):
    # img = cv2.imread('static/feed.jpg')
    with open(os.path.join(settings.BASE_DIR, 'static', 'context.json')) as json_file:
        context = json.load(json_file)
    template = loader.get_template('visionit/index.html')
    return render(request, 'visionit/index.html', context=context)


def rename(request):
    return "This is working"
