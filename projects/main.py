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

import subprocess
import cv2
import argparse
#from det2_api import hello as detectron2hello
from det2_api import drawboundingboxes as det2_dboxes
from ssd_mobilenet.api import drawboundingboxes as ssd_dboxes
#from yolov8.api import hello as yolohello
import time

#variavel global para mudar o intervalo de espera por frame
TIME_WAIT_KEY = 10

#faz o parse do argv (os argumentos que vao para a shell)
def parse(argv):
    parser = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i", "--input", dest="input", action="store", type=str,
                        help="input video file")
    return parser.parse_args(argv)

def draw_bboxes(frame, ids, confidences, boxes):
    for id_, confidence, bbox in zip(ids, confidences, boxes):
        print(id_, confidence, bbox)
        xi, yi, xf, yf = bbox
        #garante que todas as coordenadas sejam inteiras
        xi, yi, xf, yf = int(xi), int(yi), int(xf), int(yf)
        p1 = (xi, yi)
        p2 = (xf, yf)
        newFrame = cv2.rectangle(frame, p1, p2, (255, 0, 0), 4)
        center = (int(xf - ((xf - xi) / 2)), int(yf - ((yf - yi) / 2)))
        print(center)
        newFrame = cv2.circle(newFrame, center, 4, (255, 255, 255), -1)

        cv2.imshow('window-name', newFrame)
    else:
        cv2.imshow('window-name', frame)

def _execDet2(videopath):
    # Loading video
    cap = cv2.VideoCapture(videopath)

    #frame a frame
    count, start_time = 0, time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        ids, confidences, boxes, new_frame = det2_dboxes(frame, count)
        draw_bboxes(frame, ids, confidences, boxes)
        # cv2.imshow('window-name', new_frame.get_image()[:, :, ::-1])
        #cv2.imshow('window-name', frame)
        count = count + 1
        if cv2.waitKey(TIME_WAIT_KEY) & 0xFF == ord('q'):
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
        print(count/(time.time() - start_time))
    print("Exec time meta: %s seconds " % (time.time() - start_time))
    cap.release()
    cv2.destroyAllWindows()

def _execSSDMobile(videopath):
    # Loading video
    cap = cv2.VideoCapture(videopath)

    #frame a frame
    count, start_time = 0, time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        ids, confidences, boxes = ssd_dboxes(frame, count)
        draw_bboxes(frame, ids, confidences, boxes)
        count = count + 1
        if cv2.waitKey(TIME_WAIT_KEY) & 0xFF == ord('q'):
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
        print(count/(time.time() - start_time))

    print("Exec time ssd: %s seconds " % (time.time() - start_time))
    cap.release()
    cv2.destroyAllWindows()

def _execYolo(videopath):
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX('yolov8n.onnx')
    cap = cv2.VideoCapture(videopath)

    # font = cv2.FONT_HERSHEY_PLAIN
    # frame_id = 0

    #frame a frame
    start_time = time.time()
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
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
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
        print(class_ids, scores, boxes)
        cv2.imshow('window-name', frame)
        count = count + 1
        if cv2.waitKey(TIME_WAIT_KEY) & 0xFF == ord('q'):
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
        print(count/(time.time() - start_time))

    print("Exec time yolo: %s seconds " % (time.time() - start_time))
    cap.release()
    cv2.destroyAllWindows()

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

    _execSSDMobile(real_path)
    #_execDet2(real_path)
    #_execYolo(real_path)
    return 0

#0=tudo ok 1=erro
returncode = main()
sys.exit(returncode)
