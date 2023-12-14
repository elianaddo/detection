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
from ssd_mobilenet.api import drawboundingboxes as ssd_dboxes, CFG as CFG1
from yolo_api import yoloApi
#from yolov8.api import hello as yolohello
from centroid import Centroid, CentroidTracker, CrossedLine
import time

#variavel global para mudar o intervalo de espera por frame
TIME_WAIT_KEY = 10
COUNTER = 0
DENTRO = 0
FORA = 0

#faz o parse do argv (os argumentos que vao para a shell)"--webcam"
def parse(argv):
    parser = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", dest="input", action="store", type=str,
                        help="input video file")
    parser.add_argument("-w", "--webcam", dest="webcam", nargs='?', const=0, type=int, default=None,
                        help="use webcam as input. Specify the webcam index.")
    parser.add_argument('--ipcam', type=str, help='IP address of the IP camera', default=None, action="store")

    parser.add_argument("--ssd", dest="ssd", action="store_true")
    parser.add_argument("--det2", dest="det2", action="store_true")
    parser.add_argument("--yolo", dest="yolo", action="store_true")

    parser.add_argument('--c1x', type=float, help='X-coordinate of the first point (percentage)', default=0.0, action="store")
    parser.add_argument('--c1y', type=float, help='Y-coordinate of the first point (percentage)', default=60.0, action="store")
    parser.add_argument('--c2x', type=float, help='X-coordinate of the second point (percentage)', default=100.0, action="store")
    parser.add_argument('--c2y', type=float, help='Y-coordinate of the second point (percentage)', default=60.0, action="store")

    parser.add_argument('--confidence', type=float, help='Confidence threshold', default=0.4, action="store")

    parser.add_argument('--norma', type=float, help='Confidence threshold', default=12.0, action="store")

    # coordenadas do ponto de entrada
    parser.add_argument('--Rx', type=float, help='X-coordinate of the entry point (percentage)', default=0.0, action="store")
    parser.add_argument('--Ry', type=float, help='Y-coordinate of the entry point (percentage)', default=0.0, action="store")

    return parser.parse_args(argv)


def draw_bboxes(frame, ids, confidences, boxes):
    global COUNTER
    for id_, confidence, bbox in zip(ids, confidences, boxes):
        xi, yi, xf, yf = map(int, bbox)
        p1, p2 = (xi, yi), (xf, yf)
        frame = cv2.rectangle(frame, p1, p2, (255, 0, 0), 4)
        # print(bbox)
    return frame

def execute_detection(vs, detection_function, c1, c2, r, norma):
    DENTRO, FORA = 0, 0
    # vs = cv2.VideoCapture(vs)
    frame_width = int(vs.get(3))  # Get the width of the frame
    frame_height = int(vs.get(4))  # Get the height of the frame
    c1 = (float(c1[0]), float(c1[1]))
    c2 = (float(c2[0]), float(c2[1]))
    c1 = (int(c1[0] * frame_width / 100), int(c1[1] * frame_height / 100))
    c2 = (int(c2[0] * frame_width / 100), int(c2[1] * frame_height / 100))
    r = (int(r[0] * frame_width / 100), int(r[1] * frame_height / 100))
    c_tracker = CentroidTracker(max_norma=(norma / 100) * min(frame_width, frame_height))
    count, start_time = 0, time.time()
    while True:
        ret, frame = vs.read()
        if not ret:
            break
        ids, confidences, boxes = detection_function(frame, count)
        centroids = c_tracker.update(boxes)
        draw_bboxes(frame, ids, confidences, boxes)
        for centroid in centroids:
            centroid.draw(frame)
            ans = centroid.check_crossed_line((c1[0], c1[1], c2[0], c2[1]),r )
            if ans == CrossedLine.ENTERED:
                DENTRO+=1
                print("Dentro: ", DENTRO, "Fora: ", FORA, "Total: ", DENTRO - FORA)
            elif ans == CrossedLine.LEAVING:
                FORA+=1
                print("Dentro: ", DENTRO, "Fora: ", FORA, "Total: ", DENTRO - FORA)
        count += 1
        if cv2.waitKey(TIME_WAIT_KEY) & 0xFF == ord('q'): #or vs.get(cv2.CAP_PROP_POS_FRAMES) == vs.get(cv2.CAP_PROP_FRAME_COUNT):
            break
        frame = limite(frame, c1, c2)
        cv2.imshow('window-name', frame)

        # print(count / (time.time() - start_time))
    print(f"Exec time: {time.time() - start_time} seconds")
    vs.release()
    cv2.destroyAllWindows()

def _execDet2(videopath, c1, c2, r, confidence, norma):
    from det2_api import drawboundingboxes as det2_dboxes, CFG as CFG2
    CFG2["confidence"] = confidence
    execute_detection(videopath, det2_dboxes, c1, c2, r, norma)

def _execSSDMobile(videopath, c1, c2, r, confidence, norma):
    CFG1["confidence"] = confidence
    execute_detection(videopath, ssd_dboxes, c1, c2, r, norma)

def _execYolo(videopath, c1, c2, r, confidence, norma):
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX('yolov8n.onnx')
    execute_detection(videopath, lambda frame, count: yoloApi(model, frame, count, confidence), c1, c2, r, norma)

def limite(frame, c1, c2):
    cv2.line(frame ,c1, c2, (255, 255, 0), 2)
    return frame

# verifica se o video é passado - verifca na shell
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse(argv)
    video_file = args.input

    if video_file:
        if not os.path.exists(video_file):
            print("file does not exist")
            return 1
        real_path = os.path.realpath(video_file)
        vs = cv2.VideoCapture(real_path)

    if args.webcam is not None:
        if args.webcam == "":
            print("Error: Please provide a valid webcam index or use --ipcam to specify an IP camera.")
            sys.exit(1)
        else:
            # Usar o índice fornecido
            try:
                webcam_index = int(args.webcam)
                vs = cv2.VideoCapture(webcam_index)
            except ValueError:
                print("Invalid webcam index. Please provide a valid integer or use --ipcam to specify an IP camera.")
                sys.exit(1)

    elif args.ipcam is not None:
            # Usar a câmera IP
            try:
                vs = cv2.VideoCapture(args.ipcam)
            except Exception as e:
                print(f"Error opening IP camera: {e}")
                sys.exit(1)
    else:
        print("Webcam not found")
        sys.exit(1)

    c1 = (args.c1x, args.c1y)
    c2 = (args.c2x, args.c2y)

    r = (args.Rx, args.Ry)

    confidence = args.confidence
    norma = args.norma

    if args.ssd:
        _execSSDMobile(vs, c1, c2, r, confidence, norma)

    if args.det2:
        _execDet2(vs, c1, c2, r, confidence, norma)

    if args.yolo:
        _execYolo(vs, c1, c2, r, confidence, norma)

    return 0

# python main.py --modelo --confianca -norma(em % de pixeis) --p1x 0 --p1y 0 --p2x 100 --p2y 100
returncode = main()
sys.exit(returncode)
