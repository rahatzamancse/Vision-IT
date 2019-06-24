import json
import operator
import pickle
import time
from pprint import pprint
import cv2
import face_recognition
from textblob import TextBlob
from gtts import gTTS
import subprocess
import os
from model.yolo_tiny import Yolo
from util import *
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
ANALYZE = 2
LEARN = 3
GPIO.setup(ANALYZE, GPIO.IN)
GPIO.setup(LEARN, GPIO.IN)

camera = cv2.VideoCapture(0)
model = Yolo()

def get_button_input():
    """
    :return: (str) Command for server
    analyze
    learn [name]
    learn auto
    """

    thresh = 5
    analyze_cnt = 0
    learn_cnt = 0
    while True:
        analyze_val = GPIO.input(ANALYZE)
        learn_val = GPIO.input(LEARN)
        if analyze_val == 1:
            analyze_cnt += 1
            learn_cnt = 0
        elif learn_val == 1:
            learn_cnt += 1
            analyze_cnt = 0
        else:
            learn_cnt = 0
            analyze_cnt = 0
        if analyze_cnt >= thresh:
            return 'analyze'
        if learn_cnt >= thresh:
            return 'learn'

def get_img_from_cam():
    _, img = camera.read()
    img = cv2.resize(img, (932, 500))
    return img


def speak(txt, lang, offline):
    if offline:
        os.system("espeak-ng '" + txt + "'")
    else:
        tts = gTTS(txt, lang)
        tts.save('speak.mp3')
        os.system("mpg123 speak.mp3")


def translate(txt, lang):
    return str(TextBlob(txt).translate(to=lang))


with open('encoded_faces.dat', 'rb') as f:
    known_name_encodings = pickle.load(f)

while True:
    command = get_button_input()
    if command == 'exit':
        break

    if command == 'analyze':
        print('reading image...')
        img = get_img_from_cam()

        center = (img.shape[1] // 2, img.shape[0] // 2)
        print('analyzing...')
        res = model.predict(img)
        print('analyzing done')

        boxes = res['boxes']
        classes = res['classes']
        confidences = res['confidences']
        class_ids = res['class_ids']
        colors = [get_color_list(len(model.classes))[id].tolist() for id in class_ids]
        print('face recognizing')

        for i, (cls, box) in enumerate(zip(classes, boxes)):
            img_cropped = img[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])]
            # cv2.imshow('sdaf', img_cropped)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if cls != 'person':
                continue

            # print(box)
            # cur_face = img[int(box[0]):int(box[0] + box[2]), int(box[1]):int(box[1] + box[3])]
            cur_face = img[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])]

            cur_face_encoding = face_recognition.face_encodings(cur_face)
            if len(cur_face_encoding) > 0:
                cur_face_encoding = cur_face_encoding[0]
            else:
                break

            face_dists = {a:10 for a in known_name_encodings.keys()}
            for known_name, known_face_encoding in known_name_encodings.items():
                # res = face_recognition.compare_faces(known_single_face_encodings, cur_face_encoding, tolerance=0.2)
                face_dists[known_name] = face_recognition.face_distance(known_face_encoding, cur_face_encoding)

            # print(face_dists)

            # print(face_dists)
            match = None
            name, face_dist = max(face_dists.items(), key=operator.itemgetter(1))
            if face_dist <= 0.6:
                classes[i] = name

        print('face recognition done')

        img_drawed = img.copy()
        draw_prediction(img_drawed, classes, colors, np.round(confidences, 2), boxes)

        if boxes != []:
            object_centers = np.array(boxes)
            object_centers = object_centers[:, [0, 1]] + (object_centers[:, [2, 3]] // 2)
            # object_centers /= np.array([img.shape[1], img.shape[0]])
            # dists = object_centers - 0.5
            dists = object_centers - np.array([img.shape[1] // 2, img.shape[0] // 2])
            x_dists = dists[:, 0]
            y_dists = dists[:, 1]
            dists = np.sqrt(x_dists * x_dists + y_dists * y_dists)

            sorted_classes = [x for _, x in sorted(zip(dists, classes))]
            sorted_boxes = [x for _, x in sorted(zip(dists, boxes))]

            attention_img = img[
                            int(sorted_boxes[0][1]):int(sorted_boxes[0][1] + sorted_boxes[0][3]),
                            int(sorted_boxes[0][0]):int(sorted_boxes[0][0] + sorted_boxes[0][2])
                            ]

            center_threshold = 0.2 * img.shape[1]
            y_threshold = 0.25 * img.shape[0]

            # cv2.circle(img_drawed, (img.shape[1] // 2, img.shape[0] // 2), int(center_threshold), (0, 255, 0), 3)

            # for x_dist, y_dist in zip(x_dists, y_dists):
            #     cv2.line(img_drawed, center, (center[0] + int(x_dist), center[1] + int(y_dist)), (255, 0, 0), 2)

            up_left_objects = {}
            up_right_obejcts = {}
            down_left_obejcts = {}
            down_right_obejcts = {}
            center_objects = {}
            center_right_objects = {}
            center_left_objects = {}

            for object, dist, x_dist, y_dist in zip(classes, dists, x_dists, y_dists):
                # center
                if dist <= center_threshold:
                    if object in center_objects:
                        center_objects[object] += 1
                    else:
                        center_objects[object] = 1
                # left
                elif x_dist < 0:
                    # center
                    if (-y_threshold) <= y_dist <= y_threshold:
                        if object in center_left_objects:
                            center_left_objects[object] += 1
                        else:
                            center_left_objects[object] = 1
                    # up
                    elif y_dist < 0:
                        if object in up_left_objects:
                            up_left_objects[object] += 1
                        else:
                            up_left_objects[object] = 1
                    # down
                    else:
                        if object in down_left_obejcts:
                            down_left_obejcts[object] += 1
                        else:
                            down_left_obejcts[object] = 1
                # right
                else:
                    # center
                    if (-y_threshold) <= y_dist <= y_threshold:
                        if object in center_right_objects:
                            center_right_objects[object] += 1
                        else:
                            center_right_objects[object] = 1
                    # up
                    elif y_dist < 0:
                        if object in up_right_obejcts:
                            up_right_obejcts[object] += 1
                        else:
                            up_right_obejcts[object] = 1
                    # down
                    else:
                        if object in down_right_obejcts:
                            down_right_obejcts[object] += 1
                        else:
                            down_right_obejcts[object] = 1

            positional_desc = get_positional_speech(up_left_objects, up_right_obejcts, down_left_obejcts,
                                                    down_right_obejcts,
                                                    center_objects, center_right_objects, center_left_objects)

            attention_txt = ""
            if sorted_classes[0] == 'person':
                attention_txt = "Attention is to an unknown person"
            else:
                attention_txt = "Attention is to " + sorted_classes[0]

            cv2.imwrite('../monitor/static/attention_img.jpg', attention_img)

        else:
            attention_txt = 'Nothing to pay attention to.'
            positional_desc = 'There is nothing to describe.'

        context = {
            'center_positional_desc': attention_txt,
            'attention': attention_txt,
            'sonar_dist': "2.8 meters",
            'activity_desc': "We are working hard in this. <Coming SOON>",
            'positional_desc': positional_desc
        }
        cv2.imwrite('../monitor/static/processed.jpg', img_drawed)
        with open('../monitor/static/context.json', 'w') as f:
            json.dump(context, f)

        pprint(context)
        lang = 'bn'
        positional_desc_bn = translate(positional_desc, lang)
        speak(positional_desc_bn, lang, False)

    elif command == 'learn':
        n = 1
        if len(known_name_encodings) > 0:
            n = len(known_name_encodings)

        name = 'person' + str(n)

        img = get_img_from_cam()

        center = (img.shape[1] // 2, img.shape[0] // 2)
        res = model.predict(img)


        boxes = res['boxes']
        classes = res['classes']

        object_centers = np.array(boxes)
        object_centers = object_centers[:, [0, 1]] + (object_centers[:, [2, 3]] // 2)
        dists = object_centers - np.array([img.shape[0] // 2, img.shape[1] // 2])
        x_dists = dists[:, 0]
        y_dists = dists[:, 1]
        dists = np.sqrt(x_dists * x_dists + y_dists * y_dists)

        sorted_classes = [x for _, x in sorted(zip(dists, classes))]
        sorted_boxes = [x for _, x in sorted(zip(dists, boxes))]

        img_person = None
        for cls, box in zip(sorted_classes, sorted_boxes):
            if cls != 'person':
                continue
            img_person = img[int(box[0]):int(box[0] + box[2]), int(box[1]):int(box[1] + box[3])]

            break

        cv2.imshow('person', img_person)
        cv2.waitKey()
        cv2.destroyAllWindows()

        if img_person is not None:
            face_encoded = face_recognition.face_encodings(img_person)
            if len(face_encoded) <= 0:
                continue
            else:
                face_encoded = face_encoded[0]
            known_name_encodings[name] = face_encoded

            with open('encoded_faces.dat', 'wb') as f:
                print('new person saved')
                known_name_encodings = pickle.dump(known_name_encodings, f)

            cv2.imwrite('../monitor/static/' + name + '.jpg', img_person)

