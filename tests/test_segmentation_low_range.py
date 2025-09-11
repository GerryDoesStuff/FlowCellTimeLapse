import numpy as np
import sys
from pathlib import Path

# Ensure the application package is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.segmentation import segment


def test_adaptive_low_range_returns_zero_mask():
    img = np.full((10, 10), 42, dtype=np.uint8)
    mask = segment(img, method="adaptive", invert=False)
    assert np.count_nonzero(mask) == 0


def test_local_low_range_returns_zero_mask():
    img = np.full((10, 10), 42, dtype=np.uint8)
    mask = segment(img, method="local", invert=False)
    assert np.count_nonzero(mask) == 0
