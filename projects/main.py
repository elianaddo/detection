import os
import sys
import pathlib
import numpy as np

script = pathlib.Path(__file__).resolve()
project_dir = script.parent.absolute()
ssd_dir = project_dir / "ssd_mobilenet"
det2_dir = project_dir / "detectron2"
sys.path.insert(0, str(ssd_dir))
sys.path.insert(1, str(det2_dir))

import cv2
import argparse
from det2_api import drawboundingboxes as det2_dboxes
from ssd_mobilenet.api import drawboundingboxes as ssd_dboxes
#from yolov8.api import hello as yolohello
from centroid import Centroid, CentroidTracker
import time

#variavel global para mudar o intervalo de espera por frame
TIME_WAIT_KEY = 10
MAX_NORMA = 30
COUNTER = 0

CLASSES = {0: u'background',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'}

#faz o parse do argv (os argumentos que vao para a shell)
def parse(argv):
    parser = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", dest="input", action="store", type=str,
                        help="input video file")

    parser.add_argument("--ssd", dest="ssd", action="store_true")

    parser.add_argument("--det2", dest="det2", action="store_true")

    parser.add_argument("--yolo", dest="yolo", action="store_true")

    return parser.parse_args(argv)

def draw_bboxes(frame, c_tracker, ids, confidences, boxes):
    global COUNTER
    for id_, confidence, bbox in zip(ids, confidences, boxes):
        if CLASSES[id_ + 1] != "person":
            cv2.imshow('window-name', frame)
            continue
        print("draw_bboxes", CLASSES[id_ + 1], confidence, bbox)
        xi, yi, xf, yf = map(int, bbox)
        p1, p2 = (xi, yi), (xf, yf)
        frame = cv2.rectangle(frame, p1, p2, (255, 0, 0), 4)
        x, y = (int(xf - ((xf - xi) / 2)), int(yf - ((yf - yi) / 2)))
        centroid = c_tracker.update(x, y)
        frame = centroid.draw(frame)
    return frame

def execute_detection(videopath, detection_function):
    cap = cv2.VideoCapture(videopath)
    c_tracker = CentroidTracker()
    count, start_time = 0, time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        ids, confidences, boxes = detection_function(frame, count)
        frame = draw_bboxes(frame, c_tracker, ids, confidences, boxes)
        count += 1
        if cv2.waitKey(TIME_WAIT_KEY) & 0xFF == ord('q') or cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
        frame = limite(frame)
        cv2.imshow('window-name', frame)
        print(count / (time.time() - start_time))
    print(f"Exec time: {time.time() - start_time} seconds")
    cap.release()
    cv2.destroyAllWindows()

def _execDet2(videopath):
    execute_detection(videopath, det2_dboxes)

def _execSSDMobile(videopath):
    execute_detection(videopath, ssd_dboxes)

def yoloApi(model, frame, count):
    [height, width, _] = frame.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame
    scale = length / 640
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)
    outputs = model.forward()
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.5:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)
    return class_ids, scores, boxes

def _execYolo(videopath):
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX('yolov8n.onnx')
    execute_detection(videopath, lambda frame, count: yoloApi(model, frame, count))


def limite(frame, c1=(0, 0), c2=(200, 200)):
    cv2.line(frame ,c1, c2, (0, 255, 0), 9)
    return frame

# verifica se o video é passado - verifca na shell
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse(argv)
    video_file = args.input

    if video_file is None:
        print(f"Please specify a file with --input")
        return 1
    if not os.path.exists(video_file):
        print("file does not exist")
        return 1

    real_path = os.path.realpath(video_file)

    if args.ssd:
        _execSSDMobile(real_path)

    if args.det2:
        _execDet2(real_path)

    if args.yolo:
        _execYolo(real_path)

    return 0

# python main.py --modelo x --confianca x --p1x 0 --p1y 0 --p2x 100 --p3x 100
#0=tudo ok 1=erro
returncode = main()
sys.exit(returncode)
