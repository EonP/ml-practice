"""Stage 2 (Day 2): a per-pixel scikit-learn baseline.

Treat each pixel as one sample with a handful of hand-made features, and fit a
random forest. It is a real baseline for your project, and it drills the
sklearn API (fit / predict, Pipeline, grouped cross-validation, metrics).
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_FEATURES = 8


def pixel_features(img):
    """float image (H, W, 3) in [0, 1] -> (H * W, 8) feature matrix, one row per
    pixel in row-major order. Columns:
      0-2  the RGB values
      3-5  the RGB values blurred with scipy.ndimage.gaussian_filter, sigma=2,
           per channel (sigma=(2, 2, 0) blurs space but not channels)
      6-7  the row and column coordinates, scaled to [0, 1]
    """
    raise NotImplementedError("TODO: pixel_features")


def build_pixel_dataset(images, masks, n_per_image, rng):
    """Sample `n_per_image` random pixels (without replacement) from each image.
    Returns X (n_images * n_per_image, 8), y (same,), groups (same,) where groups
    holds the index of the image each pixel came from (for grouped splitting)."""
    raise NotImplementedError("TODO: build_pixel_dataset")


def make_baseline():
    """A Pipeline: StandardScaler -> RandomForestClassifier(n_estimators=100,
    max_depth=12, n_jobs=-1, random_state=0)."""
    raise NotImplementedError("TODO: make_baseline")


def grouped_cv_f1(X, y, groups, n_splits=3):
    """Cross-validate a fresh `make_baseline()` with GroupKFold so no image is
    in both train and test. Returns the list of macro-F1 scores, one per fold."""
    raise NotImplementedError("TODO: grouped_cv_f1")


def predict_mask(model, img):
    """Predict a full (H, W) label map for one image with a fitted model."""
    raise NotImplementedError("TODO: predict_mask")
