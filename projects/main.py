import os
import sys
import pathlib

script = pathlib.Path(__file__).resolve()
project_dir = script.parent.absolute()
ssd_dir = project_dir / "ssd_mobilenet"
det2_dir = project_dir / "detectron2"
sys.path.insert(0, str(ssd_dir))
sys.path.insert(1, str(det2_dir))

import subprocess
import cv2
import argparse
from det2_api import hello as detectron2hello
from det2_api import drawboundingboxes as det2_dboxes
from ssd_mobilenet.api import drawboundingboxes as ssd_dboxes
from yolov8.api import hello as yolohello
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

def _execDet2(videopath):
    start_time = time.time()
    print(detectron2hello())

    # Loading video
    cap = cv2.VideoCapture(videopath)
    # font = cv2.FONT_HERSHEY_PLAIN
    # frame_id = 0

    #frame a frame
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        class_ids, confidence, bounding_boxes, new_frame = det2_dboxes(frame, count)
        print(class_ids, confidence, bounding_boxes)
        cv2.imshow('window-name', new_frame.get_image()[:, :, ::-1])
        count = count + 1
        if cv2.waitKey(TIME_WAIT_KEY) & 0xFF == ord('q'):
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Exec time meta: %s seconds " % (time.time() - start_time))

def _execSSDMobile(videopath):
    start_time = time.time()

    # Loading video
    cap = cv2.VideoCapture(videopath)
    # font = cv2.FONT_HERSHEY_PLAIN
    # frame_id = 0

    #frame a frame

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        ids, confid, boxes = ssd_dboxes(frame, count)
        print(ids, confid, boxes)
        cv2.imshow('window-name', frame)
        count = count + 1
        if cv2.waitKey(TIME_WAIT_KEY) & 0xFF == ord('q'):
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exec time ssd: %s seconds " % (time.time() - start_time))

def _execYolo(videopath):
    start_time = time.time()
    print("Exec time yolo: %s seconds " % (time.time() - start_time))

# def execDet2(video_file):
#     start_time = time.time()
#     delta = time.time() - start_time
#     print("Exec time det: %s seconds " % round(delta, 4))
#
# def execSSDMobile(video_file):
#     start_time = time.time()
#     cwd = os.getcwd()
#     pathToGo = os.path.join(cwd, "ssd_mobilenet")
#     print(video_file)
#     print(pathToGo)
#     os.chdir(pathToGo)
#     try:
#          subprocess.run(
#              [
#                  "python",
#                  "people_counter.py",
#                  "--prototxt", "detector/MobileNetSSD_deploy.prototxt",
#                  "--model", "detector/MobileNetSSD_deploy.caffemodel",
#                  "--input", video_file
#              ])
#     except:
#         print("ocorreu um erro")
#     finally:
#         os.chdir(cwd)
#     #python people_counter.py --prototxt detector/MobileNetSSD_deploy.prototxt --model detector/MobileNetSSD_deploy.caffemodel --input utils/data/tests/test_1.mp4
#     delta = time.time() - start_time
#     print("Exec time SSD: %s seconds " % round(delta, 4))
#
# def execYolo(video_file):
#     start_time = time.time()
#     delta = time.time() - start_time
#     print("Exec time yoloV8: %s seconds " % round(delta, 4))

#verifica se o video é passado - verifca na shell
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

#    _execSSDMobile(real_path)
    _execDet2(real_path)
#    _execYolo(real_path)
    return 0

#0=tudo ok 1=erro
returncode = main()
sys.exit(returncode)
