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

def hello():
    return "Hello Meta!"

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
    return class_ids, confidence, bounding_boxes
