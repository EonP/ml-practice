import numpy as np
import torch
from torchvision.ops import box_iou as tv_box_iou
from torchvision.ops import nms as tv_nms

from drills.d12_nms import box_area, box_iou, nms

rng = np.random.default_rng(0)


def random_boxes(n):
    xy = rng.uniform(0, 50, size=(n, 2))
    wh = rng.uniform(5, 30, size=(n, 2))
    return np.c_[xy, xy + wh]


def test_area():
    np.testing.assert_allclose(box_area(np.array([[0, 0, 2, 3], [1, 1, 2, 2.0]])), [6, 1])


def test_iou_matches_torchvision():
    a, b = random_boxes(10), random_boxes(7)
    np.testing.assert_allclose(box_iou(a, b), tv_box_iou(torch.tensor(a), torch.tensor(b)).numpy(), atol=1e-10)


def test_disjoint_boxes_have_zero_iou():
    assert box_iou(np.array([[0, 0, 1, 1.0]]), np.array([[2, 2, 3, 3.0]]))[0, 0] == 0


def test_nms_matches_torchvision():
    for thr in (0.3, 0.5, 0.7):
        boxes, scores = random_boxes(60), rng.random(60)
        ref = tv_nms(torch.tensor(boxes), torch.tensor(scores), thr).numpy()
        np.testing.assert_array_equal(nms(boxes, scores, thr), ref)
