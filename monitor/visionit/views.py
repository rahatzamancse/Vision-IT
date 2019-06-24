import os
import pickle

from django.shortcuts import render
from django.template import loader
import json

from monitor import settings


def index(request):
    with open(os.path.join(settings.BASE_DIR, 'static', 'context.json')) as json_file:
        context = json.load(json_file)
    return render(request, 'visionit/index.html', context=context)


def rename(request):
    context = {
        'persons': []
    }
    for file in os.listdir('static/persons'):
        name = ''.join(file.split('.')[:-1])
        context['persons'].append(name)
    return render(request, 'visionit/rename.html', context=context)


def change_name(request, prev_name):
    new_name = request.GET.get('name')

    os.rename('static/persons/' + prev_name + '.jpg', 'static/persons/' + new_name + '.jpg')
    # with open('/home/pi/Vision-IT/project/encoded_faces.dat', 'rb') as f:
    with open('../project/encoded_faces.dat', 'rb') as f:
        data = pickle.load(f)

    print(data)
    with open('../project/encoded_faces.dat', 'wb') as f:
        if prev_name in data:
            print('renaming name')
            encoded = data[prev_name]
            del data[prev_name]
            data[new_name] = encoded
        print(data)
        pickle.dump(data, f)

    context = {
        'persons': []
    }

    for file in os.listdir('static/persons'):
        name = ''.join(file.split('.')[:-1])
        context['persons'].append(name)
    return render(request, 'visionit/rename.html', context=context)
