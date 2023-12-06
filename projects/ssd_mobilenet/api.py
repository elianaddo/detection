import time

from tracker.centroidtracker import CentroidTracker
import numpy as np
import imutils
import dlib
import cv2
import os
import pathlib

#python people_counter.py --prototxt detector/MobileNetSSD_deploy.prototxt --model detector/MobileNetSSD_deploy.caffemodel

#dicionário com a config

script = pathlib.Path(__file__).resolve()
ssd_dir = str(script.parent.absolute())
CFG = {
    "prototxt": os.path.join(ssd_dir, "detector", "MobileNetSSD_deploy.prototxt"),
    "model": os.path.join(ssd_dir, "detector", "MobileNetSSD_deploy.caffemodel"),
    "confidence" : 0.4
}

NUM_SKIP_FRAMES = 1

def hello():
    return "Hello 2"

def drawboundingboxes(frame, totalFrames):
    # array mapeia int ids para string ids
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(CFG["prototxt"], CFG["model"])

    H, W, _ = frame.shape
    scale_x = 500 / W
    scale_y = 500 / H

    frame = imutils.resize(frame, width = 500)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    rects = []

    # convert the frame to a blob and pass the blob through the
    # network and obtain the detections
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    net.setInput(blob)
    detections = net.forward()

    ids = []
    confid = []
    boxes = []

    # loop over the detections
    #CONFIDENCE
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated
        # with the prediction
        model_confidence = detections[0, 0, i, 2]

        # filter out weak detections by requiring a minimum
        # confidence
        if model_confidence > CFG["confidence"]:
            # extract the index of the class label from the
            # detections list
            idx = int(detections[0, 0, i, 1])

            #CLASSID
            # if the class label is not a person, ignore it
            if CLASSES[idx] != "person":
                continue

            # compute the (x, y)-coordinates of the bounding box
            # for the object
            #BOXXX
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            startX *= scale_x
            startY *= scale_y
            endX *= scale_x
            endY *= scale_y

            ids.append(CLASSES[idx])
            confid.append(model_confidence)
            boxes.append(box)


    return ids, confid, boxes
