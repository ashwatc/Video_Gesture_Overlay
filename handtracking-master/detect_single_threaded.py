from utils import detector_utils as detector_utils
import cv2
import tensorflow.compat.v1 as tf
import numpy as np
import datetime
import argparse

detection_graph, sess = detector_utils.load_inference_graph()

def draw_frames(cap, n, apply_overlay=lambda f:f):
    for i in range(n):
        ret, frame = cap.read()
        cv2.imshow('Camera Video', frame)
    return frame

from torchvision import transforms, models
import torch
model = models.mobilenet_v2()
model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(1280, 29))
model.load_state_dict(torch.load('gesturenet_weights'))
model.eval()
idx2gesture = ['fist', 'palm', 'other', 'ok', 'other', 'fist', 'ok', 'other',
    'other', 'other', 'other', 'peace', 'finger', 'other', 'other', 'other',
    'ok', 'ok', 'other', 'other', 'fist', 'other', 'fist', 'finger', 'peace',
    'palm', 'other', 'peace', 'finger']
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
def classify(img): #expects 128*128*3 input
    model_input = torch.unsqueeze(transform(img), dim=0)
    return idx2gesture[model(model_input).argmax()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=1,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=640,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=480,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    im_width, im_height = (cap.get(3), cap.get(4))

    cv2.namedWindow('Camera Video', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)

    GESTURE_MEMORY_LENGTH = 5
    AFK_COOLDOWN = 20
    gesture_memory = ['other']*GESTURE_MEMORY_LENGTH
    afk_countdown = AFK_COOLDOWN

    while True:
        image_bgr = draw_frames(cap, 1)
        image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        overlay_canvas = np.ones(image_np.shape, np.uint8) * np.array([255, 0, 0], np.uint8)
        face_seen = detector_utils.face_in_frame(image_np, overlay_canvas, True)
        if face_seen:
            afk_countdown = AFK_COOLDOWN
        elif afk_countdown > 0:
            afk_countdown -= 1

        if afk_countdown == 0:
            detector_utils.draw_overlay_image('afk', overlay_canvas)
        else:
            boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
            box, score = boxes[0], scores[0]
            if score > args.score_thresh:
                box = detector_utils.adjust_bounding_box(box, im_width, im_height)
                pred = detector_utils.draw_box_and_classify(box, image_bgr, classify,
                    im_width, im_height, overlay_canvas, True)
                gesture_memory.append(pred)
                gesture_memory = gesture_memory[1:]

        #decide what to draw
        if afk_countdown == 0:
            detector_utils.draw_overlay_image('afk', overlay_canvas)
        else:
            def count_then_recency(ges):
                if ges == 'other':
                    return 0
                c = 0
                for i, g in enumerate(gesture_memory):
                    if g == ges:
                        c += 1 + i*0.01
                return c
            current_gesture = max(gesture_memory, key=count_then_recency)
            detector_utils.draw_overlay_image(current_gesture, overlay_canvas)

        cv2.imshow('Overlay', overlay_canvas)
    
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
