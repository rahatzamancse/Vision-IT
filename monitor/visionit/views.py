import os
import pickle

from django.shortcuts import render
from django.template import loader
import json
import os
from django.http import JsonResponse

from monitor import settings


def index(request):
    if os.path.exists('/home/pi/busy'):
        with open(os.path.join(settings.BASE_DIR, 'static', 'context.json')) as json_file:
            context = json.load(json_file)
        return render(request, 'visionit/index.html', context=context)
    else:
        return render(request, 'visionit/index.html', context={})

def is_busy(request):
    with open('/home/pi/state', 'r') as f:
        state = f.read()
    if state == 'Ready':
        with open('/home/pi/state', 'w') as f:
            f.write('Ready1')
    data = {
        'state': state
    }
    return JsonResponse(data)


def rename(request):
    context = {
        'persons': []
    }
    for file in os.listdir('static/persons'):
        name = ''.join(file.split('.')[:-1])
        context['persons'].append(name)
    return render(request, 'visionit/rename.html', context=context)

def lang_detect(request, lang):
    with open('/home/pi/lang', 'w') as f:
        f.write(lang)
    return render(request, 'visionit/index.html', context={})

def change_name(request, prev_name):
    new_name = request.GET.get('name')

    os.rename('static/persons/' + prev_name + '.jpg', 'static/persons/' + new_name + '.jpg')
    with open('/home/pi/encoded_faces.dat', 'rb') as f:
        data = pickle.load(f)

    with open('/home/pi/encoded_faces.dat', 'wb') as f:
        if prev_name in data:
            encoded = data[prev_name]
            del data[prev_name]
            data[new_name] = encoded
        pickle.dump(data, f)

    context = {
        'persons': []
    }

    for file in os.listdir('static/persons'):
        name = ''.join(file.split('.')[:-1])
        context['persons'].append(name)
    return render(request, 'visionit/rename.html', context=context)


def delete_name(request, name):
    os.remove('static/persons/' + name + '.jpg')
    with open('/home/pi/encoded_faces.dat', 'rb') as f:
        data = pickle.load(f)

    with open('/home/pi/encoded_faces.dat', 'wb') as f:
        if name in data:
            del data[name]
        pickle.dump(data, f)

    context = {
        'persons': []
    }

    for file in os.listdir('static/persons'):
        name = ''.join(file.split('.')[:-1])
        context['persons'].append(name)
    return render(request, 'visionit/rename.html', context=context)

def map(request):
    return render(request, 'visionit/map.html')
