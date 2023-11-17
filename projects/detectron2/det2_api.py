import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import os
import pathlib
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg

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

script = pathlib.Path(__file__).resolve()
det2_dir = str(script.parent.absolute())
CFG = {
    # configs/COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml
    "config_file": os.path.join(det2_dir, "configs", "COCO-Detection", "retinanet_R_50_FPN_1x.yaml"),
    # MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_50_DC5_3x/137849425/model_final_68d202.pkl
    # https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl
    # https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_1x/190397773/model_final_bfca0b.pkl

    "opts": ["MODEL.WEIGHTS", "detectron2://COCO-Detection/retinanet_R_50_FPN_1x/190397773/model_final_bfca0b.pkl"],
    "confidence" : 0.5
}

def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cuda'
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(CFG["config_file"])
    cfg.merge_from_list(CFG["opts"])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CFG["confidence"]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CFG["confidence"]
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CFG["confidence"]
    cfg.freeze()
    return cfg

class VisualizationDemo:
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

DET2CONFIG = setup_cfg()

DEMO = VisualizationDemo(DET2CONFIG)

def drawboundingboxes(frame, totalFrames):
    preds, new_frame = DEMO.run_on_image(frame)
    confidence = preds["instances"].scores.cpu().numpy()
    class_ids = preds["instances"].pred_classes.cpu().numpy()
    bounding_boxes = preds["instances"].pred_boxes.tensor.cpu().numpy()
    # print(preds["instances"].pred_classes.tolist())
    ids=[]
    confid=[]
    bbox=[]
    for id_, confid_, bbox_ in zip(class_ids, confidence, bounding_boxes):
        if CLASSES[id_ + 1] != "person":
            continue
        ids.append(id_)
        confid.append(confid_)
        bbox.append(bbox_)
    return ids, confid, bbox
