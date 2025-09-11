import numpy as np
import sys
from pathlib import Path

# Ensure the application package is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.segmentation import segment


def test_adaptive_diff_nearly_uniform_returns_zero():
    img = np.full((20, 20), 5, dtype=np.uint8)
    mask = segment(
        img,
        method="adaptive",
        invert=False,
        use_diff=True,
    )
    assert np.count_nonzero(mask) == 0
