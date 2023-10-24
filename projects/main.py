import sys
import cv2
import argparse
from detectron2.api import hello as detectron2hello
from ssd_mobilenet.api import hello as ssdhello
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

def execDet2(videopath):
    start_time = time.time()
    print(detectron2hello())

    # Loading video
    cap = cv2.VideoCapture(videopath)
    font = cv2.FONT_HERSHEY_PLAIN
    frame_id = 0

    #frame a frame

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        #class_ids, boxes, confidences=drawboundingboxes(frame)
        #print(class_ids, boxes, confidences)
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
    print("Exec time meta: %s seconds " % (time.time() - start_time))

def ssdMobile(videopath):
    start_time = time.time()
    print(ssdhello())

    # Loading video
    cap = cv2.VideoCapture(videopath)
    font = cv2.FONT_HERSHEY_PLAIN
    frame_id = 0

    #frame a frame

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        #class_ids, boxes, confidences=drawboundingboxes(frame)
        #print(class_ids, boxes, confidences)
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

#def yolo():
#    start_time = time.time()
#    print(yolohello())
#    print("Exec time yolo: %s seconds " % (time.time() - start_time))


#verifica se o video é passado - verifca na shell
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse(argv)
    video_file = args.input

    if video_file is None:
        print(f"Please specify a file with --input")
        return 1

    print(video_file)
    execDet2(video_file)
    ssdMobile(video_file)
    return 0

#0=tudo ok 1=erro
returncode = main()
sys.exit(returncode)
