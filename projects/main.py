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
from det2_api import drawboundingboxes as det2_dboxes, CFG as CFG2
from ssd_mobilenet.api import drawboundingboxes as ssd_dboxes, CFG as CFG1
#from yolov8.api import hello as yolohello
from centroid import Centroid, CentroidTracker, CrossedLine
import time

#variavel global para mudar o intervalo de espera por frame
TIME_WAIT_KEY = 10
COUNTER = 0
DENTRO = 0
FORA = 0


CLASSES = {
    0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
    7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'stop sign',
    13: 'parking meter', 14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow',
    21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack', 26: 'umbrella', 27: 'handbag',
    28: 'tie', 29: 'suitcase', 30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite',
    35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket',
    40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife', 45: 'spoon', 46: 'bowl', 47: 'banana',
    48: 'apple', 49: 'sandwich', 50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza',
    55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch', 59: 'potted plant', 60: 'bed', 61: 'dining table',
    62: 'toilet', 63: 'tv', 64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone',
    69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book', 75: 'clock',
    76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier', 80: 'toothbrush'
}

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

    parser.add_argument('--c1x', type=int, help='X-coordinate of the first point', default=0, action="store")
    parser.add_argument('--c1y', type=int, help='Y-coordinate of the first point', default=170, action="store")
    parser.add_argument('--c2x', type=int, help='X-coordinate of the second point', default=500, action="store")
    parser.add_argument('--c2y', type=int, help='Y-coordinate of the second point', default=170, action="store")

    parser.add_argument('--confidence', type=float, help='Confidence threshold', default=0.4, action="store")

    parser.add_argument('--norma', type=int, help='Confidence threshold', default=30, action="store")

    return parser.parse_args(argv)


def draw_bboxes(frame, ids, confidences, boxes):
    global COUNTER
    for id_, confidence, bbox in zip(ids, confidences, boxes):
        if id_ != "person":
            continue
        xi, yi, xf, yf = map(int, bbox)
        p1, p2 = (xi, yi), (xf, yf)
        frame = cv2.rectangle(frame, p1, p2, (255, 0, 0), 4)
    return frame


def execute_detection(videopath, detection_function, c1, c2, norma):
    DENTRO, FORA = 0, 0
    cap = cv2.VideoCapture(videopath)
    c_tracker = CentroidTracker(max_norma=norma)
    count, start_time = 0, time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        ids, confidences, boxes = detection_function(frame, count)
        centroids = c_tracker.update(boxes)
        draw_bboxes(frame, ids, confidences, boxes)
        for centroid in centroids:
            centroid.draw(frame)
            ans = centroid.check_crossed_line((c1[0], c1[1], c2[0], c2[1]))
            if ans == CrossedLine.ENTERED:
                DENTRO+=1
                print("Dentro: ", DENTRO, "Fora: ", FORA, "Total: ", DENTRO - FORA)
            elif ans == CrossedLine.LEAVING:
                FORA+=1
                print("Dentro: ", DENTRO, "Fora: ", FORA, "Total: ", DENTRO - FORA)
        count += 1
        if cv2.waitKey(TIME_WAIT_KEY) & 0xFF == ord('q') or cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
        frame = limite(frame, c1, c2)
        cv2.imshow('window-name', frame)
        # print(count / (time.time() - start_time))
    print(f"Exec time: {time.time() - start_time} seconds")
    cap.release()
    cv2.destroyAllWindows()

def _execDet2(videopath, c1, c2, confidence, norma):
    CFG2["confidence"] = confidence
    def ddet2_dboxes_wrapper(frame, count):
        ids, confidences, boxes = det2_dboxes(frame, count)
        return map(lambda id: CLASSES[id + 1], ids), confidences, boxes
    execute_detection(videopath, ddet2_dboxes_wrapper, c1, c2, norma)

def _execSSDMobile(videopath, c1, c2, confidence, norma):
    CFG1["confidence"] = confidence
    execute_detection(videopath, ssd_dboxes, c1, c2, norma)


def yoloApi(model, frame, count, confidence):
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
        if maxScore >= confidence:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)
    return class_ids, scores, boxes

def _execYolo(videopath, c1, c2, confidence, norma):
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX('yolov8n.onnx')
    execute_detection(videopath, lambda frame, count: yoloApi(model, frame, count, confidence), c1, c2, norma)

def limite(frame, c1, c2):
    cv2.line(frame ,c1, c2, (255, 255, 0), 2)
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

    c1 = (args.c1x, args.c1y) if args.c1x is not None and args.c1y is not None else (0, 170)
    c2 = (args.c2x, args.c2y) if args.c2x is not None and args.c2y is not None else (500, 170)

    confidence = args.confidence
    norma = args.norma

    if args.ssd:
        _execSSDMobile(real_path, c1, c2, confidence, norma)

    if args.det2:
        _execDet2(real_path, c1, c2, confidence, norma)

    if args.yolo:
        _execYolo(real_path, c1, c2, confidence, norma)

    return 0

# python main.py --modelo --confianca -norma(em % de pixeis) --p1x 0 --p1y 0 --p2x 100 --p2y 100
returncode = main()
sys.exit(returncode)
