import numpy as np
from sklearn.metrics import confusion_matrix as sk_cm
from sklearn.metrics import jaccard_score, f1_score

from drills.d11_seg_metrics import confusion_matrix, dice_per_class, iou_per_class, mean_iou

rng = np.random.default_rng(0)
K = 4
target = rng.integers(0, K, size=(2, 16, 16))
pred = np.where(rng.random((2, 16, 16)) < 0.7, target, rng.integers(0, K, size=(2, 16, 16)))


def test_confusion_matrix_matches_sklearn():
    np.testing.assert_array_equal(confusion_matrix(pred, target, K), sk_cm(target.ravel(), pred.ravel(), labels=range(K)))


def test_ignore_index():
    t = target.copy()
    t[0] = 255
    cm = confusion_matrix(pred, t, K, ignore_index=255)
    assert cm.sum() == target[1].size
    np.testing.assert_array_equal(cm, sk_cm(target[1].ravel(), pred[1].ravel(), labels=range(K)))


def test_iou_matches_sklearn():
    ref = jaccard_score(target.ravel(), pred.ravel(), average=None, labels=range(K))
    np.testing.assert_allclose(iou_per_class(pred, target, K), ref)
    np.testing.assert_allclose(mean_iou(pred, target, K), ref.mean())


def test_absent_class_is_nan_and_ignored_in_mean():
    t = np.array([[0, 0], [1, 1]])
    p = np.array([[0, 0], [1, 0]])
    iou = iou_per_class(p, t, 3)
    assert np.isnan(iou[2])
    np.testing.assert_allclose(iou[:2], [2 / 3, 1 / 2])
    np.testing.assert_allclose(mean_iou(p, t, 3), (2 / 3 + 1 / 2) / 2)


def test_dice_equals_f1():
    ref = f1_score(target.ravel(), pred.ravel(), average=None, labels=range(K))
    np.testing.assert_allclose(dice_per_class(pred, target, K), ref)
