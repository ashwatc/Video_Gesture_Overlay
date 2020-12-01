from utils import detector_utils as detector_utils
import cv2
import tensorflow.compat.v1 as tf
import datetime
import argparse

detection_graph, sess = detector_utils.load_inference_graph()

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
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=240,
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

    #tf.config.experimental.set_per_process_memory_fraction(0.75)
    #tf.config.experimental.set_per_process_memory_growth(True)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    #set up gesture classifier
    from torchvision import transforms, models
    import torch
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(1280, 29))
    model.load_state_dict(torch.load('gesturenet_weights'))
    model.eval()
    classes = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
        'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space',
        'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    def classify_gesture(img): #expects 128 * 128 bgr image
        model_input = torch.unsqueeze(transform(img), dim=0)
        return classes[model(model_input).argmax()]

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np_bgr = cap.read()
        try:
            # image_np = cv2.flip(image_np, 1)
            image_np = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

        # draw bounding boxes on frame
        # detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
        #                                  scores, boxes, im_width, im_height,
        #                                  image_np)


        detector_utils.draw_box_on_image(1, args.score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np, classify_gesture)


        # gesture_image = cv2.imread("overlay-icons/question.png")
        # gesture_image = cv2.resize(gesture_image, image_np.shape[0], image_np.shape[1])
        # image_np = cv2.addWeighted(image_np, 1, gesture_image, 0.5, 0)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))
