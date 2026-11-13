"""Drill 12: box IoU and non-maximum suppression (target: 25 min). NumPy only.

Boxes are (x1, y1, x2, y2) with x2 > x1 and y2 > y1.
"""
import numpy as np


def box_area(boxes):
    """(N, 4) -> (N,)"""
    raise NotImplementedError("TODO: box_area")


def box_iou(a, b):
    """Pairwise IoU between (N, 4) and (M, 4) boxes -> (N, M). No loops (broadcast)."""
    raise NotImplementedError("TODO: box_iou")


def nms(boxes, scores, iou_threshold=0.5):
    """Greedy NMS: repeatedly keep the highest-scoring remaining box and drop
    every remaining box whose IoU with it is > iou_threshold.
    Returns the kept indices (into the original arrays), sorted by score descending."""
    raise NotImplementedError("TODO: nms")
