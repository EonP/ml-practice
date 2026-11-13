import pytest

from seg.toy import make_toy_dataset


@pytest.fixture
def toy_data():
    """12 toy (image, mask) pairs of size 32: background 0, red disc 1, blue square 2."""
    return make_toy_dataset(12, size=32, seed=0)
