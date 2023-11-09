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

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame = imutils.resize(frame, width = 500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % NUM_SKIP_FRAMES != 0:
        return [], [], []

    # convert the frame to a blob and pass the blob through the
    # network and obtain the detections
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    net.setInput(blob)
    detections = net.forward()

    ids = []
    confid = []
    boxes = []

    # loop over the detections
    # CONFIDENCE
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated
        # with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by requiring a minimum
        # confidence
        if confidence > CFG["confidence"]:
            # extract the index of the class label from the
            # detections list
            idx = int(detections[0, 0, i, 1])

            # CLASSID
            # if the class label is not a person, ignore it
            if CLASSES[idx] != "person":
                continue

            # compute the (x, y)-coordinates of the bounding box
            # for the object
            # BOXXX
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            ids.append(CLASSES[idx])
            confid.append(confidence)
            dx = (endX - startX)/2
            dy = (endY - startY)/2
            boxes.append((startX - dx, startY - dy, endX - dx, endY - dy))


    return ids, confid, boxes
