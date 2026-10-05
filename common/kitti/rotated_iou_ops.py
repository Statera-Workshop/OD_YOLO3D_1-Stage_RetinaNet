"""
Rotated (BEV) IoU helpers used inside the Complex-YOLO loss (anchor matching,
IoU metrics). Copied from Complex_Yolo_Yolov3/utils/utils.py (the CPU-safe
version); the function bodies are unchanged.

Box format everywhere: [x, y, w, l, im, re]  (im = sin(yaw), re = cos(yaw)).
"""


def to_cpu(tensor):
    return tensor.detach().cpu()
