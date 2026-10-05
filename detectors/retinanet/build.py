"""
RetinaNet builder (Image task; KITTI complex / yolo3d track: backbone + neck with the YOLOv3 layers).

Paper: Lin, Goyal, Girshick, He, Dollar, "Focal Loss for Dense Object Detection" (ICCV 2017, arXiv 1708.02002).
Code : detectron2 (facebookresearch, the authors' group; the original Detectron is Caffe2), approved as the
       official-equivalent source on 2026-10-04. Backbone classes from torchvision 0.20.1 (resnet.py).

    backbone  : ResNet-50 (res3, res4, res5)                         resnet.py   (torchvision)
    neck      : FPN P3-P5 + P6/P7 from C5, 256 channels              fpn.py      (detectron2)
    head      : 4 x (3x3 conv + ReLU) cls / box subnets, A = 9        retinanet_head.py
    anchors   : 3 scales x 3 ratios, sizes 32-512 on strides 8-128    box_ops.py
    matching  : IoU >= 0.5 fg, < 0.4 bg, ignore between, low-quality matches allowed
    loss      : focal loss (alpha 0.25, gamma 2) + L1 on deltas / EMA(#fg)   loss.py

Raw contract: [cx, cy, w, h, objectness, class probabilities]. RetinaNet has no objectness branch, so
objectness = 1 (same as v6 / v8 / YOLOX2) and the shared score = sigmoid class probability.

[Unified] differences from detectron2, all from the common protocol:
    no ImageNet weights, trainable BatchNorm, common weights_init_normal (conv weights N(0, 0.02); the
    head's prior bias -log(99) is kept because the common init does not touch biases);
    no pixel mean / std normalisation (same input as every framework detector);
    608 x 608 input, Adam 1e-4, effective batch 8 instead of SGD 0.01, batch 16, 800-pixel shorter side;
    post-processing by the common pipeline (conf 0.01, pre-NMS top-300, NMS 0.5, max 100) instead of
    score > 0.05, top-1000 per level, per-class NMS 0.5.
"""

import torch.nn as nn

from detectors.retinanet.resnet import build_resnet

VARIANT = "retinanet_r50"
RETINANET_CFG = {
    "retinanet_r50":  {"depth": 50,  "fpn_channels": 256, "num_convs": 4, "prior_prob": 0.01},
    "retinanet_r101": {"depth": 101, "fpn_channels": 256, "num_convs": 4, "prior_prob": 0.01},
}


class RetinaNetKitti(nn.Module):
    """KITTI track: ResNet-50 + FPN (P3-P5, no P6 / P7) feeding the YOLOv3 Complex-YOLO / YOLO3D layers."""

    def __init__(self, task, num_classes, variant=VARIANT):
        super().__init__()
        from detectors.retinanet.fpn import FPN
        from detectors.retinanet.kitti import KittiV3Head
        cfg = RETINANET_CFG[variant]
        self.backbone = build_resnet(cfg["depth"])
        self.fpn = FPN(self.backbone.out_channels, cfg["fpn_channels"], top_block=None)
        c = cfg["fpn_channels"]
        self.head = KittiV3Head(task, num_classes, {8: c, 16: c, 32: c}, self._features)
        self.yolo_layers = self.head.yolo_layers
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16

    def _features(self, x):
        p3, p4, p5 = self.fpn(self.backbone(x))
        return {8: p3, 16: p4, 32: p5}

    def forward(self, x, targets=None):
        return self.head(x, targets)

    def load_darknet_weights(self, path):
        raise NotImplementedError("RetinaNet has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    if task in ("complex", "yolo3d"):
        return RetinaNetKitti(task, num_classes)
    raise ValueError(f"unknown task {task}")
