import numpy as np
from pydub import AudioSegment
import socket
from gtts import gTTS
from termcolor import colored
import RPi.GPIO as GPIO
from textblob import TextBlob
import os
from googletrans import Translator
import pickle

translator = Translator()

# Translate
def translate(txt, lang):
    return translator.translate(txt, dest=lang).text

# 2 button input
def get_button_input(ANALYZE, LEARN):
    ANALYZE_ACTIVE = 0
    LEARN_ACTIVE = 0
    # a = input('command :')
    # return a
    thresh = 5
    analyze_cnt = 0
    learn_cnt = 0
    highlight('Waiting for input...')
    while True:
        analyze_val = GPIO.input(ANALYZE)
        learn_val = GPIO.input(LEARN)
        if analyze_val == ANALYZE_ACTIVE:
            analyze_cnt += 1
            learn_cnt = 0
        elif learn_val == LEARN_ACTIVE:
            learn_cnt += 1
            analyze_cnt = 0
        else:
            learn_cnt = 0
            analyze_cnt = 0
        if analyze_cnt >= thresh:
            highlight('Got input analyze')
            return 'analyze'
        if learn_cnt >= thresh:
            highlight('Got input learn')
            return 'learn'

def highlight(txt):
    print(colored('\n\n\n\n' + str(txt), 'red', 'on_cyan'))

def get_positional_speech(up_left_objects, up_right_objects, down_left_objects, down_right_objects, center_objects,
                          center_right_objects, center_left_objects, lang='en'):
    '''
    en:
        In front, there [is/are] 3 person[s], 2 table[s], [1/a] dog and 2 hair dryers.
        On the right, you get 5 bananas, 2 apples and [a/1] cake.
        Your left have a suitcase, 2 beds, [an] umbrella.
        On the top right you get 2 apples.
        Bottom right has 10 praptis.
        Upper left has 2 apples.
        Bottom left has 10 praptis.
    bn:
        আপনার সামনে 3 [টি/জন] ব্যক্তি, ২ টি টেবিল, ১ টি কুকুর এবং ২ টি চুল ড্রায়ার রয়েছে।
        ডানদিকে 5 টি কলা, ২ টি আপেল এবং ১ টি কেক আছে।
        আপনার বামে একটি স্যুটকেস, ২ টি শয্যা, ১ টি ছাতা রয়েছে।
        উপরের ডানদিকে 2 টি আপেল পান।
        নীচে ডানদিকে 10 টি প্রম্পতি রয়েছে।
        উপরের বামে 2 টি আপেল রয়েছে।
        নীচে বামদিকে 10 টি প্রম্পতি রয়েছে।
    '''

    with open('encoded_faces.dat', 'rb') as f:
        known_name_encodings = pickle.load(f)

    name_sounds = {}
    for known_name in known_name_encodings.keys():
        os.system('espeak-ng -w tmp.wav ' + known_name)
        name_sounds[known_name] = AudioSegment.from_mp3('tmp.wav')

    if lang == 'en':
        with open(os.path.join('language_dataset', 'english', 'en.pickle'), 'rb') as f:
            dictionary = pickle.load(f)
        dictionary.update(name_sounds)

        # Center
        center_positional_desc = dictionary["In front, there"] if len(center_objects) > 0 else dictionary["There is nothing interesting in front."]
        for value in center_objects.values():
            if value > 1:
                center_positional_desc += dictionary["are"]
            else:
                center_positional_desc += dictionary["is"]
            break

        c = 1
        for object, n in center_objects.items():
            if n == 1:
                n = np.random.choice(['1', 'a'])
                if n == 'a' and object[0] in ['a', 'e', 'i', 'o', 'u']:
                    n = 'an'
            else:
                n = str(n)
            # else:
                # object += 's'
            
            center_positional_desc += dictionary[n] + dictionary[object]
            # if c == len(center_objects):
                # center_positional_desc += '.'
            # else:
                # center_positional_desc += ', '
            c += 1
        # if center_positional_desc == "In front, there ":
            # center_positional_desc = "There is nothing interesting in front."


        # Center right
        center_right_positional_desc = dictionary["On the right, you get"] if len(center_right_objects) > 0 else AudioSegment.silent(1)
        c = 1
        for object, n in center_right_objects.items():
            if n == 1:
                n = np.random.choice(['1', 'a'])
                if n == 'a' and object[0] in ['a', 'e', 'i', 'o', 'u']:
                    n = 'an'
            else:
                n = str(n)
            # else:
                # object += 's'
            center_right_positional_desc += dictionary[n] + dictionary[object]
            # if c == len(center_right_objects):
                # center_right_positional_desc += '.'
            # else:
                # center_right_positional_desc += ', '
            c += 1
        # if center_right_positional_desc == "On the right, you get ":
            # center_right_positional_desc = ""

        # Center Left
        center_left_positional_desc = dictionary["Your left have"] if len(center_left_objects) > 0 else AudioSegment.silent(1)
        c = 1
        for object, n in center_left_objects.items():
            if n == 1:
                n = np.random.choice(['1', 'a'])
                if n == 'a' and object[0] in ['a', 'e', 'i', 'o', 'u']:
                    n = 'an'
            else:
                n = str(n)
            # else:
                # object += 's'
            center_left_positional_desc += dictionary[n] + dictionary[object]
            # if c == len(center_left_objects):
            #     center_left_positional_desc += '.'
            # else:
            #     center_left_positional_desc += ', '
            c += 1
        # if center_left_positional_desc == "Your left have ":
            # center_left_positional_desc = ""

        # Upper Right
        up_right_positional_desc = dictionary["On the top right, you get"] if len(up_right_objects) > 0 else AudioSegment.silent(1)
        c = 1
        for object, n in up_right_objects.items():
            if n == 1:
                n = np.random.choice(['1', 'a'])
                if n == 'a' and object[0] in ['a', 'e', 'i', 'o', 'u']:
                    n = 'an'
            else:
                n = str(n)
            # else:
                # object += 's'
            up_right_positional_desc += dictionary[n] + dictionary[object]
            # if c == len(up_right_objects):
            #     up_right_positional_desc += '.'
            # else:
            #     up_right_positional_desc += ', '
            c += 1
        # if up_right_positional_desc == "On the top right, you get ":
            # up_right_positional_desc = ""

        # Bottom Right
        down_right_positional_desc = dictionary["Bottom right have"] if len(down_right_objects) > 0 else AudioSegment.silent(1)
        c = 1
        for object, n in down_right_objects.items():
            if n == 1:
                n = np.random.choice(['1', 'a'])
                if n == 'a' and object[0] in ['a', 'e', 'i', 'o', 'u']:
                    n = 'an'
            else:
                n = str(n)
            # else:
                # object += 's'
            down_right_positional_desc += dictionary[n] + dictionary[object]
            # if c == len(down_right_objects):
            #     down_right_positional_desc += '.'
            # else:
            #     down_right_positional_desc += ', '
            c += 1
        # if down_right_positional_desc == "Bottom right have ":
            # down_right_positional_desc = ""

        # Upper Left
        up_left_positional_desc = dictionary["Upper left have"] if len(up_left_objects) > 0 else AudioSegment.silent(1)
        c = 1
        for object, n in up_left_objects.items():
            if n == 1:
                n = np.random.choice(['1', 'a'])
                if n == 'a' and object[0] in ['a', 'e', 'i', 'o', 'u']:
                    n = 'an'
            else:
                n = str(n)
            # else:
                # object += 's'
            up_left_positional_desc += dictionary[n] + dictionary[object]
            # if c == len(up_left_objects):
            #     up_left_positional_desc += '.'
            # else:
            #     up_left_positional_desc += ', '
            c += 1
        # if up_left_positional_desc == "Upper left have ":
        #     up_left_positional_desc = ""

        # Bottom Left
        down_left_positional_desc = dictionary["Bottom left have"] if len(down_left_objects) > 0 else AudioSegment.silent(1)
        c = 1
        for object, n in down_left_objects.items():
            if n == 1:
                n = np.random.choice(['1', 'a'])
                if n == 'a' and object[0] in ['a', 'e', 'i', 'o', 'u']:
                    n = 'an'
            else:
                n = str(n)
            # else:
            #     object += 's'
            down_left_positional_desc += dictionary[n] + dictionary[object]
            # if c == len(down_left_objects):
            #     down_left_positional_desc += '.'
            # else:
            #     down_left_positional_desc += ', '
            c += 1
        # if down_left_positional_desc == "Bottom left have ":
            # down_left_positional_desc = ""
    
    elif lang == 'bn':
        '''
        bn:
            আপনার সামনে 3 [টি/জন] ব্যক্তি, ২ টি টেবিল, ১ টি কুকুর এবং ২ টি চুল ড্রায়ার রয়েছে।
            ডানদিকে 5 টি কলা, ২ টি আপেল এবং ১ টি কেক আছে।
            আপনার বামে একটি স্যুটকেস, ২ টি শয্যা, ১ টি ছাতা রয়েছে।
            উপরের ডানদিকে 2 টি আপেল পান।
            নীচে ডানদিকে 10 টি প্রম্পতি রয়েছে।
            উপরের বামে 2 টি আপেল রয়েছে।
            নীচে বামদিকে 10 টি প্রম্পতি রয়েছে।
        '''
        with open(os.path.join('language_dataset', 'bangla', 'bn.pickle'), 'rb') as f:
            dictionary = pickle.load(f)
        dictionary.update(name_sounds)
        
        bn_num = {
            0: '০',
            1: '১',
            2: '২',
            3: '৩',
            4: '৪',
            5: '৫',
            6: '৬',
            7: '৭',
            8: '৮',
            9: '৯',
            10: '১০',
        }
        bn_object = {
            'person': 'ব্যক্তি',
            'bicycle': 'সাইকেল',
            'car': 'গাড়ী',
            'motorcycle': 'মোটরসাইকেল',
            'airplane': 'বিমান',
            'bus': 'বাস',
            'train': 'রেলগাড়ি',
            'truck': 'ট্রাক',
            'boat': 'নৌকা',
            'traffic light': 'ট্রাফিক বাতি',
            'fire hydrant': 'ফায়ার হাইড্র্যান্ট',
            'stop sign': 'সাইন বন্ধ',
            'parking meter': 'পার্কিং মিটার',
            'bench': 'এজলাস',
            'bird': 'পাখি',
            'cat': 'বিড়াল',
            'dog': 'কুকুর',
            'horse': 'ঘোড়া',
            'sheep': 'মেষ',
            'cow': 'গাভী',
            'elephant': 'হাতি',
            'bear': 'ভালুক',
            'zebra': 'জেব্রা',
            'giraffe': 'জিরাফ',
            'backpack': 'ব্যাকপ্যাক',
            'umbrella': 'ছাতা',
            'handbag': 'হ্যান্ডব্যাগ',
            'tie': 'টাই',
            'suitcase': 'সুটকেস',
            'frisbee': 'চাকতি',
            'skis': 'স্কি',
            'snowboard': 'স্নোবোর্ড',
            'sports ball': 'ক্রীড়া বল',
            'kite': 'ঘুড়ি',
            'baseball bat': 'বেসবল ব্যাট',
            'baseball glove': 'বেসবল দস্তানা',
            'skateboard': 'স্কেটবোর্ডের',
            'surfboard': 'সার্ফবোর্ড',
            'tennis racket': 'টেনিস র্যাকেট',
            'bottle': 'বোতল',
            'wine glass': 'সুরাপাত্র',
            'cup': 'কাপ',
            'fork': 'কাঁটাচামচ',
            'knife': 'ছুরি',
            'spoon': 'চামচ',
            'bowl': 'বাটি',
            'banana': 'কলা',
            'apple': 'আপেল',
            'sandwich': 'স্যান্ডউইচ',
            'orange': 'কমলা',
            'broccoli': 'ব্রোকলি',
            'carrot': 'গাজর',
            'hot dog': 'হট ডগ',
            'pizza': 'পিজা',
            'donut': 'ডোনাট',
            'cake': 'পিষ্টক',
            'chair': 'চেয়ার',
            'couch': 'পালঙ্ক',
            'potted plant': 'সংক্ষেপিত উদ্ভিদ',
            'bed': 'বিছানা',
            'dining table': 'খাবার টেবিল',
            'toilet': 'টয়লেট',
            'tv': 'টেলিভিশন',
            'laptop': 'ল্যাপটপ',
            'mouse': 'মাউস',
            'remote': 'দূরবর্তী',
            'keyboard': 'কীবোর্ড',
            'cell phone': 'মুঠোফোন',
            'microwave': 'মাইক্রোওয়েভ',
            'oven': 'চুলা',
            'toaster': 'টোস্ট করার বৈদু্যতিক যন্ত্র',
            'sink': 'ডুবা',
            'refrigerator': 'ফ্রিজ',
            'book': 'বই',
            'clock': 'ঘড়ি',
            'vase': 'দানি',
            'scissors': 'কাঁচি',
            'teddy bear': 'টেডি বিয়ার',
            'hair drier': 'সভ্রসেযাভবসেত',
            'toothbrush': 'টুথব্রাশ',
        }
        for name in name_sounds.keys():
            bn_object[name] = name

        print(bn_object)
        print()
        print()
        print()
        print()
        print(dictionary)

        # Center
        center_positional_desc = dictionary["আপনার সামনে"] if len(center_objects) > 0 else dictionary["সামনে বর্ণনা এর কিছু নেই।"]

        c = 1
        for object, n in center_objects.items():
            if c > 1 and c == len(center_objects) - 1:
                center_positional_desc += dictionary["এবং"]

            center_positional_desc += dictionary[bn_num[n]] + dictionary["টি"] + dictionary[bn_object[object]]
                

            if c == len(center_objects):
                center_positional_desc += dictionary['রয়েছে']
            # else:
            #     center_positional_desc += ', '
            c += 1
        # if center_positional_desc == "আপনার সামনে":
            # center_positional_desc = "সামনে বর্ণনা এর কিছু নেই।"


        # Center right
        center_right_positional_desc = dictionary["ডানদিকে"] if len(center_right_objects) > 0 else AudioSegment.silent(1)
        c = 1
        for object, n in center_right_objects.items():
            if c > 1 and c == len(center_right_objects) - 1:
                center_right_positional_desc += dictionary["এবং"]

            center_right_positional_desc += dictionary[bn_num[n]] + dictionary["টি"] + dictionary[bn_object[object]]

            if c == len(center_right_objects):
                center_right_positional_desc += dictionary['আছে']
            # else:
                # center_right_positional_desc += ', '
            c += 1
        # if center_right_positional_desc == "ডানদিকে ":
            # center_right_positional_desc = ""


        # Center Left
        center_left_positional_desc = dictionary["বামে"] if len(center_left_objects) > 0 else AudioSegment.silent(1)
        c = 1
        for object, n in center_left_objects.items():
            if c > 1 and c == len(center_left_objects) - 1:
                center_left_positional_desc += dictionary["এবং"]

            center_left_positional_desc += dictionary[bn_num[n]] + dictionary["টি"] + dictionary[bn_object[object]]

            if c == len(center_left_objects):
                center_left_positional_desc += dictionary['আছে']
            # else:
                # center_left_positional_desc += ', '
            c += 1
        # if center_left_positional_desc == "বামে":
            # center_left_positional_desc = ""

        # Upper Right
        up_right_positional_desc = dictionary["উপরের ডানদিকে"] if len(up_right_objects) > 0 else AudioSegment.silent(1)
        c = 1
        for object, n in up_right_objects.items():
            if c > 1 and c == len(up_right_objects) - 1:
                up_right_positional_desc += dictionary["এবং"]

            up_right_positional_desc += dictionary[bn_num[n]] + dictionary["টি"] + dictionary[bn_object[object]]

            if c == len(up_right_objects):
                up_right_positional_desc += dictionary['আছে']
            # else:
                # up_right_positional_desc += ', '
            c += 1
        # if up_right_positional_desc == "উপরের ডানদিকে ":
            # up_right_positional_desc = ""

        # Bottom Right
        down_right_positional_desc = dictionary["নীচে ডানদিকে"] if len(down_right_objects) > 0 else AudioSegment.silent(1)
        c = 1
        for object, n in down_right_objects.items():
            if c > 1 and c == len(down_right_objects) - 1:
                down_right_positional_desc += dictionary["এবং"]

            down_right_positional_desc += dictionary[bn_num[n]] + dictionary["টি"] + dictionary[bn_object[object]]

            if c == len(down_right_objects):
                down_right_positional_desc += dictionary['আছে']
            # else:
                # down_right_positional_desc += ', '
            c += 1
        # if down_right_positional_desc == "নীচে ডানদিকে":
            # down_right_positional_desc = ""

        # Upper Left
        up_left_positional_desc = dictionary["উপরের বামে"] if len(up_left_objects) > 0 else AudioSegment.silent(1)
        c = 1
        for object, n in up_left_objects.items():
            if c > 1 and c == len(up_left_objects) - 1:
                up_left_positional_desc += dictionary["এবং"]

            up_left_positional_desc += dictionary[bn_num[n]] + dictionary["টি"] + dictionary[bn_object[object]]

            if c == len(up_left_objects):
                up_left_positional_desc += dictionary['আছে']
            # else:
                # up_left_positional_desc += ', '
            c += 1
        # if up_left_positional_desc == "উপরের বামে ":
            # up_left_positional_desc = ""

        # Bottom Left
        down_left_positional_desc = dictionary["নীচে বামে"] if len(down_left_objects) > 0 else AudioSegment.silent(1)
        c = 1
        for object, n in up_left_objects.items():
            if c > 1 and c == len(up_left_objects) - 1:
                down_left_positional_desc += dictionary["এবং"]

            down_left_positional_desc += dictionary[bn_num[n]] + dictionary["টি"] + dictionary[bn_object[object]]

            if c == len(up_left_objects):
                down_left_positional_desc += dictionary['আছে']
            # else:
                # down_left_positional_desc += ', '
            c += 1
        # if down_left_positional_desc == "নীচে বামে ":
            # down_left_positional_desc = ""

    positional_desc = center_positional_desc + center_right_positional_desc + center_left_positional_desc + up_right_positional_desc + up_left_positional_desc + down_left_positional_desc + down_right_positional_desc

    positional_desc.export('speak.mp3', format="mp3")

    return positional_desc
