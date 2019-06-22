import numpy as np
import cv2


def draw_prediction(img, labels, colors, confidences, boxes):
    for i in range(len(boxes)):
        x = int(boxes[i][0])
        y = int(boxes[i][1])
        w = int(boxes[i][2])
        h = int(boxes[i][3])
        color = colors[i]

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, str(labels[i]) + " : " + str(confidences[i]), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def get_color_list(n):
    return np.random.randint(0, 255, size=(n, 3), dtype='uint8')

def get_positional_speech(up_left_objects, up_right_objects, down_left_objects, down_right_objects, center_objects,
                          center_right_objects, center_left_objects):
    '''
    In front, there [are] 3 person[s], 2 tables, [1/a] dog and 2 hair dryers.
    On the right, you get 5 bananas, 2 apples and [a/1] cake.
    Your left have a suitcase, 2 beds, [an] umbrella.
    On the top right you get 2 apples.
    Bottom right has 10 praptis.
    Upper left has 2 apples.
    Bottom left has 10 praptis.
    '''
    # Center
    center_positional_desc = "In front, there "
    for value in center_objects.values():
        if value > 1:
            center_positional_desc += "are "
        else:
            center_positional_desc += "is "
        break

    c = 1
    for object, n in center_objects.items():
        if n == 1:
            n = np.random.choice(['1', 'a'])
            if n == 'a' and object[0] in ['a', 'e', 'i', 'o', 'u']:
                n = 'an'
        else:
            object += 's'
        center_positional_desc += n + " " + object
        if c == len(center_objects):
            center_positional_desc += '.'
        else:
            center_positional_desc += ', '
        c += 1
    if center_positional_desc == "In front, there ":
        center_positional_desc = "There is nothing interesting in front."


    # Center right
    center_right_positional_desc = "On the right, you get "
    c = 1
    for object, n in center_right_objects.items():
        if n == 1:
            n = np.random.choice(['1', 'a'])
            if n == 'a' and object[0] in ['a', 'e', 'i', 'o', 'u']:
                n = 'an'
        else:
            object += 's'
        center_right_positional_desc += n + " " + object
        if c == len(center_right_objects):
            center_right_positional_desc += '.'
        else:
            center_right_positional_desc += ', '
        c += 1
    if center_right_positional_desc == "On the right, you get ":
        center_right_positional_desc = ""


    # Center Left
    center_left_positional_desc = "Your left have "
    c = 1
    for object, n in center_left_objects.items():
        if n == 1:
            n = np.random.choice(['1', 'a'])
            if n == 'a' and object[0] in ['a', 'e', 'i', 'o', 'u']:
                n = 'an'
        else:
            object += 's'
        center_left_positional_desc += n + " " + object
        if c == len(center_left_objects):
            center_left_positional_desc += '.'
        else:
            center_left_positional_desc += ', '
        c += 1
    if center_left_positional_desc == "Your left have ":
        center_left_positional_desc = ""


    # Upper Right
    up_right_positional_desc = "On the top right, you get "
    c = 1
    for object, n in up_right_objects.items():
        if n == 1:
            n = np.random.choice(['1', 'a'])
            if n == 'a' and object[0] in ['a', 'e', 'i', 'o', 'u']:
                n = 'an'
        else:
            object += 's'
        up_right_positional_desc += n + " " + object
        if c == len(up_right_objects):
            up_right_positional_desc += '.'
        else:
            up_right_positional_desc += ', '
        c += 1
    if up_right_positional_desc == "On the top right, you get ":
        up_right_positional_desc = ""


    # Bottom Right
    down_right_positional_desc = "Bottom right have "
    c = 1
    for object, n in down_right_objects.items():
        if n == 1:
            n = np.random.choice(['1', 'a'])
            if n == 'a' and object[0] in ['a', 'e', 'i', 'o', 'u']:
                n = 'an'
        else:
            object += 's'
        down_right_positional_desc += n + " " + object
        if c == len(down_right_objects):
            down_right_positional_desc += '.'
        else:
            down_right_positional_desc += ', '
        c += 1
    if down_right_positional_desc == "Bottom right have ":
        down_right_positional_desc = ""

    # Upper Left
    up_left_positional_desc = "Upper left have "
    c = 1
    for object, n in up_left_objects.items():
        if n == 1:
            n = np.random.choice(['1', 'a'])
            if n == 'a' and object[0] in ['a', 'e', 'i', 'o', 'u']:
                n = 'an'
        else:
            object += 's'
        up_left_positional_desc += n + " " + object
        if c == len(up_left_objects):
            up_left_positional_desc += '.'
        else:
            up_left_positional_desc += ', '
        c += 1
    if up_left_positional_desc == "Upper left have ":
        up_left_positional_desc = ""


    # Bottom Left
    down_left_positional_desc = "Bottom left have "
    c = 1
    for object, n in down_left_objects.items():
        if n == 1:
            n = np.random.choice(['1', 'a'])
            if n == 'a' and object[0] in ['a', 'e', 'i', 'o', 'u']:
                n = 'an'
        else:
            object += 's'
        down_left_positional_desc += n + " " + object
        if c == len(down_left_objects):
            down_left_positional_desc += '.'
        else:
            down_left_positional_desc += ', '
        c += 1
    if down_left_positional_desc == "Bottom left have ":
        down_left_positional_desc = ""

    positional_desc = center_positional_desc + " " + center_right_positional_desc + " " + center_left_positional_desc + " " + up_right_positional_desc + " " + up_left_positional_desc + " " + down_left_positional_desc + " " + down_right_positional_desc

    return positional_desc
