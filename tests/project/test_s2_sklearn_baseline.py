import numpy as np
from sklearn.pipeline import Pipeline

from seg.s2_sklearn_baseline import build_pixel_dataset, grouped_cv_f1, make_baseline, pixel_features, predict_mask


def test_pixel_features(toy_data):
    img = toy_data[0][0] / 255.0
    f = pixel_features(img)
    assert f.shape == (32 * 32, 8)
    np.testing.assert_allclose(f[:, :3], img.reshape(-1, 3))
    np.testing.assert_allclose(f[0, 6:], [0, 0])
    np.testing.assert_allclose(f[-1, 6:], [1, 1])
    np.testing.assert_allclose(f[31, 6:], [0, 1])          # end of first row
    assert f[:, 3:6].std() < f[:, :3].std()                 # blurring reduces variance


def test_build_dataset(toy_data):
    imgs = [i / 255.0 for i in toy_data[0]]
    X, y, g = build_pixel_dataset(imgs, toy_data[1], 200, np.random.default_rng(0))
    assert X.shape == (12 * 200, 8) and y.shape == g.shape == (12 * 200,)
    np.testing.assert_array_equal(np.bincount(g), [200] * 12)


def test_baseline_is_pipeline():
    assert isinstance(make_baseline(), Pipeline)


def test_baseline_learns_toy_task(toy_data):
    imgs = [i / 255.0 for i in toy_data[0]]
    X, y, g = build_pixel_dataset(imgs, toy_data[1], 300, np.random.default_rng(0))
    scores = grouped_cv_f1(X, y, g, n_splits=3)
    assert len(scores) == 3 and min(scores) > 0.9
    model = make_baseline().fit(X, y)
    pred = predict_mask(model, imgs[0])
    assert pred.shape == (32, 32)
    assert (pred == toy_data[1][0]).mean() > 0.95
