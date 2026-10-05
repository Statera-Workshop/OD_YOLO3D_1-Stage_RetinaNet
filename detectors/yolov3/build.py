"""
YOLOv3 builders for the three tasks.

Everything here is algorithm-specific:
    - which .cfg (backbone + neck + head layout)
    - how [yolo] anchors are parsed
    - which detection layer (loss / target assignment) is used
    - how many box attributes the head regresses

Everything else (Darknet parsing, training, evaluation, demo) is common.
"""

import math
from pathlib import Path

from detectors.yolov3.yolo3d_yolo_layer import YoloLayer as Yolo3dYoloLayer

CFG_DIR = Path(__file__).resolve().parent / "cfg"


def _floats(text):
    return [float(v) for v in text.split(",")]


def _mask(block):
    return [int(v) for v in block["mask"].split(",")]


def yolo3d_factory(block, hyperparams):
    """YOLO3D: (h, w, l, yaw) anchors -> (h, w, l, sin, cos) (legacy)."""
    v = _floats(block["anchors"])
    anchors = [(v[i], v[i + 1], v[i + 2], math.sin(v[i + 3]), math.cos(v[i + 3])) for i in range(0, len(v), 4)]
    return Yolo3dYoloLayer(anchors=[anchors[i] for i in _mask(block)], num_classes=int(block["classes"]),
                           stride=1, ignore_thresh=0.5)


SPECS = {
    #          cfg file                 factory          box attributes regressed
    "yolo3d":  ("yolo3d_yolov3.cfg",  yolo3d_factory,  8),   # x y z h w l yaw conf  (legacy: 9)
}
