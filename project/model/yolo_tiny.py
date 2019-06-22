import numpy as np
import cv2


class Yolo:
    def __init__(self, yolo_cfg='model/yolov3-tiny.cfg', yolo_weights='model/yolov3-tiny.weights', classes_path='model/coco_yolov3_80labels.txt'):
        self.scale = 0.00392

        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
        self.output_layers = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.conf_threshold = 0.5
        self.nms_threshold = 0.4

    def predict(self, img):
        ret = {
            'image': img,
            'class_ids': [],
            'classes': [],
            'confidences': [],
            'boxes': [],
        }
        class_ids = []
        confidences = []
        boxes = []

        height = img.shape[0]
        width = img.shape[1]

        blob = cv2.dnn.blobFromImage(img, self.scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # print(type(outs))
        # print(len(outs))
        # print(len(outs[0]))
        # print(len(outs[0][0]))
        # print(outs)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        for i in indices:
            i = i[0]
            ret['boxes'].append(boxes[i])
            ret['confidences'].append(confidences[i])
            ret['class_ids'].append(class_ids[i])
            ret['classes'].append(self.classes[class_ids[i]])
            ret['image'] = img

            # draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

        return ret
