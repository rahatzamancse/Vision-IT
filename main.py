#!/usr/bin/python3
# Import packages
import os
with open('/home/pi/lang', 'r') as f:
    lang = f.read()
if lang == 'en':
    lang_path = os.path.join('language_dataset', 'english')
elif lang == 'bn':
    lang_path = os.path.join('language_dataset', 'bangla')

os.system("mpg123 " + os.path.join(lang_path, "booting.mp3"))
with open('state','w') as f:
    f.write('Loading modules to memory...')

from util import *
highlight('Loading modules to memory...')
import time
timer_start = time.time()
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import RPi.GPIO as GPIO
import face_recognition
import operator
import pickle
import json
from termcolor import colored

timer_stop = time.time()
highlight('Modules loaded in '+ str(int(timer_stop-timer_start)) + ' Seconds')

with open('state','w') as f:
    f.write('Loading machine learning model to memory...')
timer_start = time.time()
highlight('Loading model to memory...')

# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
#IM_WIDTH = 640    Use smaller resolution for
#IM_HEIGHT = 480   slightly faster framerate

# This is needed since the working directory is the object_detection folder.
sys.path.append('/home/pi/tensorflow1/models/research/object_detection')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
# Grab path to current working directory
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,'tensorflow1','models','research','object_detection',MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'tensorflow1','models','research','object_detection','data','mscoco_label_map.pbtxt')
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# catergory_index = {id: {'id': 'name'}, ...}
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

timer_stop = time.time()
highlight('Model loaded in ' + str(int(timer_stop-timer_start)) + ' seconds')


# Initialize GPIO pins
GPIO.setmode(GPIO.BCM)
ANALYZE = 2
LEARN = 3
GPIO.setup(ANALYZE, GPIO.IN)
GPIO.setup(LEARN, GPIO.IN)


os.system("mpg123 " + os.path.join(lang_path, "ready.mp3"))

with PiCamera() as camera:
    with PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT)) as output:
        camera.resolution = (IM_WIDTH,IM_HEIGHT)
        camera.framerate = 10

        while True:
            with open('state','w') as f:
                f.write('Ready')
            command = get_button_input(ANALYZE, LEARN)
            if command == 'exit':
                break


            with open('encoded_faces.dat', 'rb') as f:
                known_name_encodings = pickle.load(f)
                highlight('Previously saved persons list :' + str(list(known_name_encodings.keys())))

            with open('/home/pi/lang', 'r') as f:
                lang = f.read()
            if lang == 'en':
                lang_path = os.path.join('language_dataset', 'english')
            elif lang == 'bn':
                lang_path = os.path.join('language_dataset', 'bangla')

            output.truncate(0)

            camera.capture(output, format='bgr')
            img = np.copy(output.array)

            # img = cv2.flip(img, 0)
            img.setflags(write=1)

            if command == 'analyze':
                with open('state','w') as f:
                    f.write('Detecting objects')
                timer_start = time.time()
                highlight('Analyzing Camera Image...')
                frame_expanded = np.expand_dims(img, axis=0)

                highlight('Detecting objects...')
                # Perform the actual detection by running the model with the image as input
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: frame_expanded})

                # maximum 100 objects
                num = int(num[0])
                # boxes = [[y, x, height, width], ...] # proportion
                boxes = np.squeeze(boxes)[:num]
                # score = [score1, ...]
                scores = np.squeeze(scores)[:num]
                # classes = [id, ...]
                classes = np.squeeze(classes).astype(np.int32)[:num]

                # num = [total_objects]

                # print('cat_ind : ', category_index)
                # print('boxes : ', boxes)
                # print('scores : ', scores)
                # print('classes : ', classes)
                # print('num : ', num)

                # Draw the results of the detection (aka 'visulaize the results')
                vis_util.visualize_boxes_and_labels_on_image_array(
                    img,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=0.40)

                # All the results have been drawn on the img, so it's time to display it.
                cv2.imwrite('/home/pi/monitor/static/processed.jpg', img)
                classes = [category_index[i]['name'] for i in classes]

                timer_stop = time.time()
                highlight('Objects detected in ' + str(int(timer_stop-timer_start)) + ' seconds')

                with open('state','w') as f:
                    f.write('Recognizing faces...')
                timer_start = time.time()
                highlight('Recognizing faces...')
                for i, cls, box in zip(range(int(num)), classes, boxes):
                    box = [int(box[0]*IM_HEIGHT), int(box[1]*IM_WIDTH), int(box[2]*IM_HEIGHT), int(box[3]*IM_WIDTH)]

                    if cls != 'person':
                        continue

                    cur_face = img[ box[0]:box[0]+box[2], box[1]:box[1]+box[3] ]

                    cur_face_encoding = face_recognition.face_encodings(cur_face)
                    if len(cur_face_encoding) > 0:
                        cur_face_encoding = cur_face_encoding[0]
                    else:
                        break

                    print(known_name_encodings.keys())
                    face_dists = face_recognition.face_distance(list(known_name_encodings.values()), cur_face_encoding)

                    min_id = face_dists.argmin()
                    name = list(known_name_encodings.keys())[min_id]

                    if min(face_dists) <= 0.6:
                        classes[i] = name

                highlight(classes)

                timer_stop = time.time()
                highlight('Face recognition done in ' + str(int(timer_stop-timer_start)) + ' seconds')

                with open('state','w') as f:
                    f.write('Generating speech...')

                up_left_objects = {}
                up_right_obejcts = {}
                down_left_obejcts = {}
                down_right_obejcts = {}
                center_objects = {}
                center_right_objects = {}
                center_left_objects = {}

                if num > 0:
                    object_centers = boxes[:, [0, 1]] + (boxes[:, [2, 3]] / 2)
                    object_centers = (object_centers * [IM_HEIGHT, IM_WIDTH]).astype(np.int32)[:num]
                    dists = object_centers - np.array([IM_HEIGHT // 2, IM_WIDTH // 2])
                    x_dists = dists[:, 0]
                    y_dists = dists[:, 1]
                    dists = np.sqrt(x_dists * x_dists + y_dists * y_dists)

                    print('Distances from objects :', dists)

                    sorted_classes = [x for _, x in sorted(zip(dists, classes))]
                    sorted_boxes = [x for _, x in sorted(zip(dists, boxes))]

                    attention_img = img[
                                    int(sorted_boxes[0][0]*IM_HEIGHT):int(sorted_boxes[0][0]*IM_HEIGHT + sorted_boxes[0][2]*IM_HEIGHT),
                                    int(sorted_boxes[0][1]*IM_WIDTH):int(sorted_boxes[0][1]*IM_WIDTH + sorted_boxes[0][3]*IM_WIDTH)
                                    ]

                    center_threshold = 0.3 * IM_WIDTH
                    y_threshold = 0.4 * IM_HEIGHT

                    # cv2.circle(img_drawed, (img.shape[1] // 2, img.shape[0] // 2), int(center_threshold), (0, 255, 0), 3)

                    # for x_dist, y_dist in zip(x_dists, y_dists):
                    #     cv2.line(img_drawed, center, (center[0] + int(x_dist), center[1] + int(y_dist)), (255, 0, 0), 2)


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
                                                        center_objects, center_right_objects, center_left_objects, lang)

                context = {
                    # 'positional_desc': positional_desc
                    'positional_desc': ""
                }
                with open('context.json', 'w') as f:
                    json.dump(context, f)

                highlight(context)
                
                os.system("mpg123 speak.mp3")

            elif command == 'learn':
                with open('state','w') as f:
                    f.write('Learning face...')
                timer_start = time.time()

                n = 1
                if len(known_name_encodings) > 0:
                    n = len(known_name_encodings)
                name = 'person' + str(n)

                frame_expanded = np.expand_dims(img, axis=0)

                highlight('Detecting objects...')
                # Perform the actual detection by running the model with the image as input
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: frame_expanded})

                # maximum 100 objects
                num = int(num[0])
                # boxes = [[y, x, height, width], ...] # proportion
                boxes = np.squeeze(boxes)[:num]
                # score = [score1, ...]
                scores = np.squeeze(scores)[:num]
                # classes = [id, ...]
                classes = np.squeeze(classes).astype(np.int32)[:num]

                # num = [total_objects]

                # print('cat_ind : ', category_index)
                # print('boxes : ', boxes)
                # print('scores : ', scores)
                # print('classes : ', classes)
                # print('num : ', num)

                classes = [category_index[i]['name'] for i in classes]

                # Draw the results of the detection (aka 'visulaize the results')
                img2store = img.copy()
                vis_util.visualize_boxes_and_labels_on_image_array(
                    img2store,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=0.40)

                # All the results have been drawn on the img, so it's time to display it.
                cv2.imwrite('/home/pi/monitor/static/processed.jpg', img2store)

                timer_stop = time.time()
                highlight('Objects detected in ' + str(int(timer_stop-timer_start)) + ' seconds')

                if boxes != []:
                    object_centers = boxes[:, [0, 1]] + (boxes[:, [2, 3]] / 2)
                    object_centers = (object_centers * [IM_HEIGHT, IM_WIDTH]).astype(np.int32)[:num]
                    dists = object_centers - np.array([IM_HEIGHT // 2, IM_WIDTH // 2])
                    x_dists = dists[:, 0]
                    y_dists = dists[:, 1]
                    dists = np.sqrt(x_dists * x_dists + y_dists * y_dists)

                    sorted_classes = [x for _, x in sorted(zip(dists, classes))]
                    sorted_boxes = [x for _, x in sorted(zip(dists, boxes))]

                    img_person = None
                    for cls, box in zip(sorted_classes, sorted_boxes):
                        if cls != 'person':
                            continue
                        img_person = img[int(box[1]*IM_HEIGHT):int(box[1]*IM_HEIGHT + box[3]*IM_HEIGHT), int(box[0]*IM_WIDTH):int(box[0]*IM_WIDTH + box[2]*IM_WIDTH)]
                        break

                    if img_person is not None:
                        face_encoded = face_recognition.face_encodings(img_person)
                        if len(face_encoded) <= 0:
                            os.system("mpg123 " + os.path.join(lang_path, "learn-fail.mp3"))
                            continue
                        else:
                            face_encoded = face_encoded[0]
                        known_name_encodings[name] = face_encoded

                        with open('encoded_faces.dat', 'wb') as f:
                            print('new person saved :', name)
                            pickle.dump(known_name_encodings, f)

                        cv2.imwrite('/home/pi/monitor/static/persons/' + name + '.jpg', img_person)
                        highlight(known_name_encodings.keys())
                        os.system("mpg123 " + os.path.join(lang_path, "learned.mp3"))
                    else:
                        os.system("mpg123 " + os.path.join(lang_path, "learn-fail.mp3"))

                else:
                    os.system("mpg123 " + os.path.join(lang_path, "learn-fail.mp3"))
