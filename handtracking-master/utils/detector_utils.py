# Utilities for object detector.

import numpy as np
import sys
import tensorflow.compat.v1 as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict


detection_graph = tf.Graph()
sys.path.append("..")

# score threshold for showing bounding boxes.
_score_thresh = 0.27

MODEL_NAME = 'hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

NUM_CLASSES = 1
# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.75
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        sess = tf.Session(graph=detection_graph)#, config=config)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np, classify):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)

            ### RESIZING BOUNDING BOX IMAGE (SQUARE + LARGE)
            width = right - left
            height = bottom - top
            if (width > height):
                diff = width - height
                bottom += diff / 2
                top -= diff / 2
            else:
                diff = height - width
                right += diff / 2
                left -= diff / 2

            width = right - left
            height = bottom - top
            right += width // 4
            left -= width // 4
            bottom += height // 4
            top -= height // 4

            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)

            ### OVERLAY PNG IMAGE OVER FRAME
            gesture_image = cv2.imread("overlay-icons/question.png")
            gesture_image = cv2.resize(gesture_image, (gesture_image.shape[1] // 4, gesture_image.shape[0] // 4))
            background_mask = np.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=np.uint8)
            background_mask[0:gesture_image.shape[0], 0:gesture_image.shape[1], :] = gesture_image
            #cv2.addWeighted(background_mask, 1, image_np, 1, 0, image_np)

            top, bottom, left, right = map(int, (top, bottom, left, right))
            #print(top, bottom, left, right)
            img_h, img_w, img_c = image_np.shape
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            tpad, bpad, lpad, rpad = max(0, -top), max(0, bottom-img_h), max(0, -left), max(0, right-img_w)
            try:
                padded = cv2.copyMakeBorder(image_bgr,
                    tpad, bpad, lpad, rpad,
                    borderType=cv2.BORDER_REPLICATE)
                hand = padded[ top-tpad:bottom-tpad, left-lpad:right-lpad, : ]
                hand = cv2.resize(hand, (128, 128))
                pred = classify(hand)
                cv2.putText(image_np, pred, (200, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
            except:
                pass

            ### IMPORT TRAINED GESTURE MODEL, AND CLASSIFICATION/OVERLAY
            # output = model.predict_classes(img)
            # cv2.draw(annotated_image)


# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    ### MODIFICATIONS:
    # boxes = np.squeeze(boxes)

    return np.squeeze(boxes), np.squeeze(scores)


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
