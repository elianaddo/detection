import sys
import argparse
from projects.detectron2.api import api
from projects.ssd_mobilenet.api import api
from projects.yolov8.api import api

def parse(argv):
    parser = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i", "--input", dest="input", action="store", type=str,
                        help="input video file")
    return parser.parse_args(argv)

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse(argv)

    video_file = args.input
    if video_file is None:
        print(f"Please specify a file with --input")
        return 1

    print(video_file)
    return 0

sys.exit(main())
