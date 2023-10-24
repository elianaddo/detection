import sys
import argparse
from detectron2.api import hello as detectron2hello
from ssd_mobilenet.api import hello as ssdhello
#from yolov8.api import hello as yolohello
import time


#faz o parse do argv (os argumentos que vao para a shell)
def parse(argv):
    parser = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i", "--input", dest="input", action="store", type=str,
                        help="input video file")
    return parser.parse_args(argv)

def execDet2():
    start_time = time.time()
    print(detectron2hello())
    print("Exec time meta: %s seconds " % (time.time() - start_time))

def ssdMobile()
    start_time = time.time()
    print(ssdhello())
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
    execDet2()
    ssdMobile()
    return 0

#0=tudo ok 1=erro
returncode = main()
sys.exit(returncode)
