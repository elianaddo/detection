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

#faz o parse do argv (os argumentos que vao para a shell)
def parse(argv):
    parser = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", dest="input", action="store", type=str,
                        help="input video file")
    return parser.parse_args(argv)

def draw_bboxes(frame, c_tracker, ids, confidences, boxes):
    global COUNTER
    for id_, confidence, bbox in zip(ids, confidences, boxes):
        print("draw_bboxes", id_, confidence, bbox)
        xi, yi, xf, yf = map(int, bbox)
        p1, p2 = (xi, yi), (xf, yf)
        newFrame = cv2.rectangle(frame, p1, p2, (255, 0, 0), 4)
        x, y = (int(xf - ((xf - xi) / 2)), int(yf - ((yf - yi) / 2)))
        centroid = c_tracker.update(x, y)
        newFrame = centroid.draw(newFrame)
        cv2.imshow('window-name', newFrame)
    else:
        cv2.imshow('window-name', frame)

def execute_detection(videopath, detection_function):
    cap = cv2.VideoCapture(videopath)
    c_tracker = CentroidTracker()
    count, start_time = 0, time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        ids, confidences, boxes = detection_function(frame, count)
        draw_bboxes(frame, c_tracker, ids, confidences, boxes)
        count += 1
        if cv2.waitKey(TIME_WAIT_KEY) & 0xFF == ord('q') or cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
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

    #_execSSDMobile(real_path)
    #_execDet2(real_path)
    _execYolo(real_path)
    return 0

#0=tudo ok 1=erro
returncode = main()
sys.exit(returncode)
