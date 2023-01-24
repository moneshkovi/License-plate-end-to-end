import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging

import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from exttext import extex


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
input_size = 416

saved_model_loaded = tf.saved_model.load('./checkpoints/custom-416', tags=[tag_constants.SERVING])


def getbboxes(image):
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.


    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)


    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]


    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.30
    )

    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
    
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    allowed_classes = list(class_names.values())

    image,bboxes = utils.draw_bbox(original_image, pred_bbox, False, allowed_classes=allowed_classes, read_plate = True)

    return image,bboxes
    

def pad(f):
    x = 0.05
    for i in range(len(f)):
        f[i][0] = int(f[i][0]*(1-x))
        f[i][1] = int(f[i][1]*(1-x))
        f[i][2] = int(f[i][2]*(1+x))
        f[i][3] = int(f[i][3]*(1+x))

    return f

if __name__=="__main__":
    i = 0

    cap = cv2.VideoCapture('rtsp://192.168.1.2:8080/out.h264')



    while cap.isOpened():

        _,img = cap.read()
        _,f = getbboxes(img)

        texts = extex(img,f)
        
        for bb,text in zip(f,texts):
            x1 = bb[0]
            y1 = bb[1]
            x2 = bb[2]
            y2 = bb[3]
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(img, text, (x1, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        # cv2.imshow("sv",img)
        # cv2.imwrite("./processed/{}.jpg".format(i),img)
        # print(i)
        # i+=1
        
        cv2.waitKey(10)
